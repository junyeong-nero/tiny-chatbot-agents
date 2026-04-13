# 평가 프레임워크

`src/evaluation/`에 구현된 평가 프레임워크는 RAG 파이프라인의 품질을 자동 메트릭과 LLM-as-a-Judge 두 계층으로 측정합니다. `main.py evaluate` CLI로 실행합니다.

---

## 실행 흐름

```
1. 데이터셋 로드
   └─ 질문, 정답, 예상 소스가 담긴 JSON 파일

2. 파이프라인 초기화 (선택)
   └─ 평가 대상 모델로 RAGPipeline 동적 생성

3. 추론
   └─ 각 테스트 케이스에 대해 답변 생성 + 컨텍스트 검색

4. 자동 메트릭 계산 (Layer 1)
   └─ BLEU, 의미론적 유사도, 환각 검증, 컨텍스트 재현율

5. LLM-as-a-Judge (Layer 2, 선택)
   └─ GPT-4o 등 고성능 모델이 1–5점 척도로 품질 평가

6. 집계 및 리포트
   └─ 카테고리별 통계 + JSON / Markdown / CSV 출력
```

---

## 핵심 컴포넌트

| 파일 | 역할 |
|------|------|
| `src/evaluation/runner.py` | 실행 오케스트레이션, 병렬화, 결과 집계 |
| `src/evaluation/evaluator.py` | 자동 메트릭 계산 구현 |
| `src/evaluation/llm_judge.py` | LLM 기반 평가 인터페이스 |
| `src/evaluation/dataset_generator.py` | 합성 평가 데이터셋 생성 도구 |
| `src/evaluation/report.py` | Markdown / CSV 리포트 생성 |

---

## 메트릭 체계

### Layer 1 — 자동 메트릭 (추가 LLM 호출 불필요)

| 메트릭 | 범위 | 설명 |
|--------|------|------|
| `answer_similarity` | 0–1 | 생성 답변과 정답의 임베딩 코사인 유사도 |
| `bleu_score` | 0–1 | N-gram 중복도; 한국어는 `kiwipiepy` 형태소 분석 적용 |
| `verifier_faithfulness` | 0–1 | `AnswerVerifier` 기반 컨텍스트 충실도 (환각 검사) |
| `context_recall` | 0–1 | 예상 소스 중 실제로 검색된 비율 |
| `context_precision` | 0–1 | 검색된 문서 중 관련 있는 문서의 비율 |
| `latency_ms` | ms | 응답 생성 시간 |
| `input_tokens` | 수 | 입력 토큰 수 |
| `output_tokens` | 수 | 출력 토큰 수 |

### Layer 2 — LLM-as-a-Judge (선택적, 1–5점 척도)

| 메트릭 | 설명 |
|--------|------|
| `llm_correctness` | 정답 대비 정확도 |
| `llm_helpfulness` | 사용자 의도 충족도 |
| `judge_context_faithfulness` | 컨텍스트와의 모순 여부 |
| `llm_fluency` | 문법적 자연스러움 |
| `llm_judge_score` | 종합 품질 점수 |

> 점수는 0–1로 정규화된 버전도 함께 제공됩니다.
>
> **권장 사용법**: 빠른 반복 개발에는 Layer 1, 최종 모델 선택 또는 심층 분석에는 Layer 2를 사용하세요.

---

## 데이터셋 형식

평가 데이터셋은 `data/evaluation/` 아래에 JSON 파일로 저장합니다.

```json
[
  {
    "question": "개인정보 처리 목적 및 항목은 무엇인가요?",
    "expected_answer": "수집 목적은 서비스 제공이며, 항목에는 이름, 이메일, 연락처가 포함됩니다.",
    "category": "개인정보처리방침",
    "expected_sources": ["privacy_section_3", "privacy_section_5"]
  }
]
```

| 필드 | 필수 여부 | 설명 |
|------|-----------|------|
| `question` | 필수 | 평가용 질문 |
| `expected_answer` | 필수 | 정답(골든 레퍼런스) |
| `category` | 선택 | 리포트의 카테고리별 집계에 사용 |
| `expected_sources` | 선택 | `context_recall` / `context_precision` 계산에 사용 |

---

## CLI 사용법

### 기본 실행

```bash
python main.py evaluate --models "llama3.1:8b"
```

### 고급 옵션

```bash
# 복수 모델 병렬 평가 + LLM-as-a-Judge + 리포트 생성
python main.py evaluate \
  --models "llama3.1:8b,mistral:7b" \
  --dataset data/evaluation/my_test_set.json \
  --use-llm-judge \
  --parallel \
  --max-workers 8 \
  --report

# 파이프라인 없이 레퍼런스 간 메트릭만 계산
python main.py evaluate --no-pipeline
```

### 전체 옵션 목록

| 옵션 | 설명 |
|------|------|
| `--models` | 쉼표 구분 모델 이름 목록 |
| `--dataset` | 평가 데이터셋 파일 경로 |
| `--provider` | 파이프라인 LLM 제공자 오버라이드 |
| `--use-llm-judge` | LLM-as-a-Judge 활성화 |
| `--judge-provider` | 판관 모델 제공자 |
| `--judge-model` | 판관 모델 이름 |
| `--auto-diverse-judge` | 다양성 편향 방지 판관 자동 선택 |
| `--strict-diversity` | 엄격한 다양성 모드 |
| `--parallel` | 병렬 실행 활성화 |
| `--max-workers` | 병렬 워커 수 |
| `--report` | `results/`에 Markdown / CSV 리포트 생성 |

---

## 편향 방지: 모델 다양성

동일 모델이 자신의 출력을 평가하면 점수가 과도하게 높게 나타나는 "홈그라운드 이점(home-field advantage)" 문제가 발생할 수 있습니다.

`--auto-diverse-judge` 옵션은 데이터셋 생성 모델과 **다른 제공자**의 판관 모델을 자동 선택합니다.

| 생성 모델 | 자동 선택 판관 |
|-----------|---------------|
| GPT-4 계열 | Claude 계열 |
| Claude 계열 | Gemini 계열 |
| 로컬 LLM | GPT-4o (설정 시) |

설정은 `configs/evaluation_config.yaml`의 `diversity_pairs`에서 관리합니다.

---

## 결과 출력

평가 완료 후 `results/` 디렉토리에 세 가지 형식으로 저장됩니다.

| 파일 | 내용 |
|------|------|
| `*.json` | 모든 테스트 케이스의 전체 트레이스 (생성 텍스트, 검색 컨텍스트, 개별 점수) |
| `*_report.md` | 카테고리별 집계 표가 포함된 Markdown 요약 |
| `*_report.csv` | Excel·데이터 분석 도구용 CSV |

---

*파이프라인 아키텍처와 연계된 동작을 이해하려면 [RAG 아키텍처](rag_architecture.md)를 참고하세요.*
