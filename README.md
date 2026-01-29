# tiny-chatbot-agents

> 약관 및 QnA 기반 CS 챗봇

QnA DB와 약관(ToS) DB를 활용한 **계층적 RAG 파이프라인**을 제공하는 프로젝트입니다.

---

## 프로젝트 개요

### 핵심 목표
- **계층적 검색 파이프라인**: QnA → 약관 → 상담원 순서의 효율적인 질문 처리
- **MCP 서버 지원**: Claude Desktop 등 MCP 클라이언트와 연동 가능
- **자동 확장 시스템**: 상담원 답변이 자동으로 QnA DB에 추가되어 지속적으로 학습

### 기술 스택
| 구분 | 기술 |
|------|------|
| **LLM (Production)** | vLLM, sglang, ollama (로컬 서빙) |
| **LLM (Testing)** | OpenAI GPT-4o-mini (테스트 전용) |
| **Embedding** | Qwen3-Embedding (0.6B/8B), multilingual-e5-large |
| **Vector DB** | ChromaDB |
| **MCP** | FastMCP |
| **Package Manager** | uv |

> **보안 참고**: 프로덕션 환경에서는 데이터 보안을 위해 외부 API 대신 로컬 LLM 서빙 프레임워크(vLLM, sglang, ollama)를 사용합니다.

---

## 시스템 아키텍처

### RAG 파이프라인

```
사용자 질문
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  [STEP 1] QnA DB 검색                                │
│  • FAQ 데이터베이스에서 유사 질문 검색                  │
│  • Similarity Score >= 0.80 → LLM context로 답변 생성 │
└─────────────────────────────────────────────────────┘
    │ (매칭 실패)
    ▼
┌─────────────────────────────────────────────────────┐
│  [STEP 2] ToS DB 검색 (Hybrid RAG)                   │
│  • Vector + Rule + Triplet 기반 하이브리드 검색        │
│  • 조항 패턴 매칭: "제1조", "제2조 1항" 등             │
│  • 키워드 부스팅: 환불, 해지, 위약금 등                 │
│  • Score >= 0.65 → LLM context로 답변 생성            │
└─────────────────────────────────────────────────────┘
    │ (검색 실패)          │ (답변 생성됨)
    ▼                      ▼
┌─────────────────────────────────────────────────────┐
│  [STEP 3] No Context 응답  │  [STEP 4] 답변 검증      │
│  • 관련 정보 없음 안내      │  • 3-Layer 검증 시스템   │
│  • 고객센터 연결 안내       │  • Hallucination 탐지    │
│                           │  • 인용 유효성 검증       │
└─────────────────────────────────────────────────────┘
                            │
                            ▼
                    검증 완료된 최종 답변
```

### 데이터베이스 구성

| DB | 용도 | 데이터 |
|----|------|--------|
| **QnA Vector DB** | FAQ 검색 | 질문-답변 쌍 |
| **ToS Vector DB** | 약관 검색 | 이용약관 조항별 Chunk |

---

## 프로젝트 구조

```
tiny-chatbot-agents/
├── data/
│   ├── raw/                    # 크롤링 원본 데이터
│   │   ├── qna/                # QnA JSON 파일
│   │   └── tos/                # 약관 JSON 파일
│   ├── evaluation/             # 평가 데이터셋
│   └── vectordb/               # ChromaDB 저장소
│       ├── qna/
│       └── tos/
├── src/
│   ├── crawlers/               # 데이터 크롤러
│   ├── vectorstore/            # Vector DB (ChromaDB)
│   │   ├── qna_store.py
│   │   ├── tos_store.py
│   │   ├── backfill.py         # 상담원 답변 Backfill
│   │   └── embeddings.py
│   ├── tos_search/             # ToS 하이브리드 검색
│   │   ├── rule_matcher.py     # 조항 패턴 + 키워드 매칭
│   │   ├── triplet_store.py    # Subject-Predicate-Object 검색
│   │   └── hybrid_search.py    # Vector + Rule + Triplet 결합
│   ├── llm/                    # LLM 클라이언트
│   │   ├── base.py             # 추상 베이스 클래스
│   │   ├── local_client.py     # vLLM/sglang/ollama 클라이언트
│   │   ├── openai_client.py    # OpenAI 클라이언트 (테스트용)
│   │   └── factory.py          # 클라이언트 팩토리
│   ├── pipeline/               # RAG 파이프라인
│   │   └── rag_pipeline.py
│   ├── mcp/                    # MCP 서버
│   │   └── server.py
│   ├── evaluation/             # 평가 시스템
│   │   ├── evaluator.py        # 기본 메트릭 평가
│   │   ├── runner.py           # 배치 평가 실행기
│   │   ├── report.py           # 리포트 생성
│   │   ├── frontier_client.py  # Claude/GPT/Gemini 클라이언트
│   │   ├── llm_judge.py        # LLM-as-a-Judge
│   │   ├── judge_prompts.py    # 평가 프롬프트
│   │   └── dataset_generator.py # 데이터셋 생성
│   └── verifier/               # Hallucination 검증
│       ├── verifier.py         # AnswerVerifier 3-Layer 검증
│       └── prompts.py          # 검증용 프롬프트 템플릿
├── scripts/
│   ├── ingest_qna.py           # QnA 데이터 적재
│   ├── ingest_tos.py           # ToS 데이터 적재
│   ├── backfill_agent_answers.py # 상담원 답변 Backfill
│   ├── run_pipeline.py         # CLI 실행
│   ├── run_mcp_server.py       # MCP 서버 실행
│   ├── run_evaluation.py       # 평가 실행
│   └── generate_dataset.py     # 평가 데이터셋 생성
├── configs/
│   ├── embedding_config.yaml
│   └── claude_desktop_config.example.json
├── results/                    # 평가 결과 출력
└── tests/
```

---

## 시작하기

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/tiny-chatbot-agents.git
cd tiny-chatbot-agents

# 의존성 설치 (uv 사용)
uv pip install -e ".[all]"

# 또는 pip 사용
pip install -e ".[all]"
```

### 2. 데이터 적재

```bash
# QnA 데이터 적재
python scripts/ingest_qna.py

# ToS 데이터 적재
python scripts/ingest_tos.py

# 특정 파일만 적재
python scripts/ingest_qna.py --file data/raw/qna/qnacrawler_xxx.json

# 기존 데이터 삭제 후 적재
python scripts/ingest_tos.py --clear

# 상담원 답변 Backfill (Human-in-the-loop)
python scripts/backfill_agent_answers.py --file data/raw/agent_answers.json

# 중복 체크 후 추가
python scripts/backfill_agent_answers.py --file answers.json --check-duplicates

# 단일 답변 추가
python scripts/backfill_agent_answers.py --add "질문" "답변" --category "계좌"

# 기존 답변 검색
python scripts/backfill_agent_answers.py --search "계좌 해지"
```

**상담원 답변 JSON 형식:**
```json
[
  {
    "question": "계좌 해지 방법이 뭐야?",
    "answer": "고객센터 또는 앱에서 해지 신청 가능합니다.",
    "category": "계좌",
    "agent_id": "agent_001",
    "session_id": "sess_12345"
  }
]
```

### 3. LLM 서버 실행

프로덕션 환경에서는 로컬 LLM 서버를 먼저 실행해야 합니다.

```bash
# vLLM 서버 (권장)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# 또는 sglang
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 30000

# 또는 ollama
ollama serve  # 기본 포트 11434
ollama pull llama3.1:8b
```

### 4. 파이프라인 실행

```bash
# LLM 프로바이더 설정 (기본값: vllm)
export LLM_PROVIDER="vllm"  # vllm, sglang, ollama, openai

# Interactive 모드
python scripts/run_pipeline.py

# 단일 질문 (vLLM 사용)
python scripts/run_pipeline.py -q "계좌 해지 방법이 뭐야?"

# 특정 프로바이더 지정
python scripts/run_pipeline.py --llm-provider ollama -q "환불 규정 알려줘"

# 특정 모델 지정
python scripts/run_pipeline.py --llm-provider ollama --llm-model mistral:7b -q "질문"

# 테스트용 OpenAI 사용 (외부 API 주의)
export OPENAI_API_KEY="sk-..."
python scripts/run_pipeline.py --llm-provider openai -q "테스트 질문"

# 검색만 테스트 (LLM 불필요)
python scripts/run_pipeline.py --search-qna "비밀번호"
python scripts/run_pipeline.py --search-tos "제1조"
```

---

## MCP 서버

MCP (Model Context Protocol) 서버를 통해 Claude Desktop 등 MCP 클라이언트와 연동할 수 있습니다.

### MCP 도구 목록

| 도구 | 설명 |
|------|------|
| `ask_question` | QnA→ToS 순서로 검색 후 LLM 답변 생성 |
| `search_faq` | FAQ(QnA) DB 직접 검색 |
| `search_terms` | 약관(ToS) DB 검색 ("제1조" 또는 키워드) |
| `get_section` | 특정 약관 조항 전체 내용 조회 |
| `list_documents` | 등록된 약관 목록 조회 |

### 실행 방법

```bash
# MCP 서버 실행
python scripts/run_mcp_server.py

# 또는
python -m src.mcp.server
```

### Claude Desktop 설정

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-chatbot": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/tiny-chatbot-agents",
      "env": {
        "LLM_PROVIDER": "vllm",
        "VLLM_API_BASE": "http://localhost:8000/v1"
      }
    }
  }
}
```

> **참고**: 테스트 목적으로 OpenAI를 사용하려면 `LLM_PROVIDER`를 `openai`로, `OPENAI_API_KEY`를 설정하세요.

---

## 설정

### LLM 프로바이더

프로덕션 환경에서는 보안을 위해 로컬 LLM 서빙 프레임워크를 사용합니다.

| 프로바이더 | 기본 Endpoint | 기본 모델 | 용도 |
|-----------|--------------|----------|------|
| **vLLM** | `http://localhost:8000/v1` | `meta-llama/Llama-3.1-8B-Instruct` | Production (권장) |
| **sglang** | `http://localhost:30000/v1` | `meta-llama/Llama-3.1-8B-Instruct` | Production |
| **ollama** | `http://localhost:11434/v1` | `llama3.1:8b` | Production / 개발 |
| **openai** | OpenAI API | `gpt-4o-mini` | **테스트 전용** |

**환경변수 설정:**
```bash
# 프로바이더 선택
export LLM_PROVIDER="vllm"  # vllm, sglang, ollama, openai

# 커스텀 endpoint (선택사항)
export VLLM_API_BASE="http://gpu-server:8000/v1"
export SGLANG_API_BASE="http://gpu-server:30000/v1"
export OLLAMA_API_BASE="http://localhost:11434/v1"

# OpenAI 테스트용 (프로덕션 사용 금지)
export OPENAI_API_KEY="sk-..."
```

**코드에서 사용:**
```python
from src.llm import create_llm_client, LLMProvider, LocalLLMClient

# 환경변수 기반 자동 선택 (기본: vllm)
client = create_llm_client()

# 명시적 프로바이더 지정
client = create_llm_client(provider=LLMProvider.OLLAMA)

# 커스텀 설정
client = LocalLLMClient(
    provider=LLMProvider.VLLM,
    model="meta-llama/Llama-3.1-70B-Instruct",
    base_url="http://gpu-server:8000/v1",
    temperature=0.5,
)

# Health check
if client.health_check():
    response = client.generate([{"role": "user", "content": "Hello"}])
```

### 임베딩 모델 (`configs/embedding_config.yaml`)

```yaml
default_model: multilingual-e5-large

models:
  multilingual-e5-large:
    name: intfloat/multilingual-e5-large
    dimension: 1024

  qwen3-embedding-0.6b:
    name: Qwen/Qwen3-Embedding-0.6B
    dimension: 1024
    max_seq_length: 32768

  qwen3-embedding-8b:
    name: Qwen/Qwen3-Embedding-8B
    dimension: 4096
    max_seq_length: 32768
```

### 파이프라인 임계값

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `qna_threshold` | 0.80 | QnA 매칭 최소 유사도 |
| `tos_threshold` | 0.65 | ToS 검색 최소 유사도 |
| `enable_hybrid_tos_search` | False | ToS 하이브리드 검색 활성화 |
| `enable_verification` | True | Hallucination 검증 활성화 |
| `verification_threshold` | 0.70 | 검증 통과 최소 신뢰도 |

### ToS 하이브리드 검색

ToS 검색에서 Embedding Vector만으로 어려운 유사도 측정을 보완합니다.

**점수 결합 공식:**
```
final_score = 0.5 × vector_score + 0.3 × rule_score + 0.2 × triplet_score
```

| 검색 방식 | 설명 |
|-----------|------|
| **Vector** | 의미론적 유사도 (ChromaDB) |
| **Rule** | 조항 패턴 (`제1조`, `제2조 1항`) + 키워드 부스팅 (환불, 해지 등) |
| **Triplet** | Subject-Predicate-Object 관계 매칭 (예: `회사-가능-환불거부`) |

**사용 예시:**
```python
from src.pipeline import RAGPipeline

# 하이브리드 검색 활성화
pipeline = RAGPipeline(enable_hybrid_tos_search=True)

# 최초 1회: Triplet 인덱스 빌드
pipeline.tos_store.build_triplet_index()

# 조항 직접 검색 (Rule 매칭)
response = pipeline.query("제3조 알려줘")

# 키워드 기반 검색 (키워드 부스팅)
response = pipeline.query("환불 규정이 어떻게 되나요?")
```

---

## Hallucination Verifier

RAG 파이프라인에서 생성된 답변의 신뢰성을 검증하는 3-Layer 방어 시스템입니다.

### 3-Layer 검증 시스템

```
생성된 답변
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  [Layer 1] Citation Check (출처 검증)                │
│  • [참조: 제N조] 형식의 인용 패턴 추출                 │
│  • 인용된 조항이 실제 컨텍스트에 존재하는지 확인        │
│  • 긴 답변(500자 이상)에 출처가 없으면 경고            │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  [Layer 2] Rule-based Check (패턴 기반 검증)         │
│  • 불확실한 표현 탐지:                                │
│    - "일반적으로", "보통", "아마도"                    │
│    - "~할 수도 있습니다", "제 생각에는", "추측컨대"    │
│  • 약관 답변에서 추측성 표현은 Hallucination 의심      │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  [Layer 3] LLM-based Check (LLM 기반 검증)           │
│  • Secondary LLM을 사용한 Faithfulness 검증          │
│  • 답변의 모든 사실이 컨텍스트에 명시적으로 존재하는가? │
│  • 컨텍스트에 없는 정보를 지어내지 않았는가?          │
│  • JSON 형식으로 검증 결과 반환                       │
└─────────────────────────────────────────────────────┘
    │
    ▼
검증 결과 (VerificationResult)
```

### 검증 결과 구조

```python
@dataclass
class VerificationResult:
    verified: bool        # 검증 통과 여부
    confidence: float     # 신뢰도 점수 (0.0 ~ 1.0)
    issues: list[str]     # 발견된 문제점 목록
    reasoning: str        # 검증 근거 설명
    citations_valid: bool # 인용 유효성
```

### 신뢰도 점수 계산

```
기본 신뢰도: 1.0

감점 요인:
- 출처 인용 문제: -0.30
- 불확실한 표현 (개당): -0.15
- LLM 검증 실패: (LLM 신뢰도 + 현재 신뢰도) / 2

최종 신뢰도 = max(0.0, min(1.0, 계산된 신뢰도))
```

### Verifier 설정 옵션

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `confidence_threshold` | 0.70 | 검증 통과 최소 신뢰도 |
| `require_citations` | True | 출처 인용 필수 여부 |
| `use_llm_verification` | True | LLM 기반 검증 사용 여부 |

### 사용 예시

**파이프라인에서 자동 검증:**
```python
from src.pipeline import RAGPipeline

# 검증 활성화 (기본값)
pipeline = RAGPipeline(
    enable_verification=True,
    verification_threshold=0.7,
)

response = pipeline.query("환불 규정이 어떻게 되나요?")

# 검증 결과 확인
print(f"검증 통과: {response.verified}")
print(f"검증 점수: {response.verification_score}")
print(f"문제점: {response.verification_issues}")
```

**직접 Verifier 사용:**
```python
from src.verifier import AnswerVerifier
from src.llm import create_llm_client

# Verifier 초기화
verifier = AnswerVerifier(
    llm_client=create_llm_client(),
    confidence_threshold=0.7,
    require_citations=True,
    use_llm_verification=True,
)

# 답변 검증
result = verifier.verify(
    question="환불이 가능한가요?",
    answer="네, 환불이 가능합니다. [참조: 제5조]",
    context=[
        {
            "section_title": "제5조 환불 규정",
            "section_content": "회원은 서비스 이용 후 7일 이내 환불을 요청할 수 있습니다."
        }
    ],
)

print(f"검증 결과: {result.verified}")
print(f"신뢰도: {result.confidence}")
print(f"근거: {result.reasoning}")
```

**빠른 검증 (LLM 없이):**
```python
# Rule-based 검증만 수행 (LLM 호출 없음)
is_valid = verifier.quick_verify(
    answer="환불은 7일 이내 가능합니다. [참조: 제5조]",
    context=[{"section_title": "제5조", "section_content": "..."}],
)
```

### 검증 시 감지되는 패턴

**Hallucination 의심 표현:**
```python
HALLUCINATION_PATTERNS = [
    r"일반적으로",      # 약관에는 일반적 설명이 아닌 구체적 규정만 있음
    r"보통",           # 추측성 표현
    r"아마도",         # 불확실성 표현
    r"~할 수도 있습니다", # 가능성 표현
    r"제 생각에는",     # 개인 의견
    r"추측컨대",       # 명시적 추측
]
```

**인용 패턴:**
```python
# 유효한 인용 형식
CITATION_PATTERN = r"\[참조:\s*([^\]]+)\]|\[출처:\s*([^\]]+)\]"

# 예시:
# [참조: 제3조 2항]
# [출처: 이용약관 제5조]
```

### 평가 시스템과의 통합

Hallucination Verifier는 평가 시스템의 Faithfulness 검증에도 활용됩니다:

```python
from src.evaluation import Evaluator

evaluator = Evaluator()

# Faithfulness 평가 시 Verifier 로직 활용
result = evaluator.evaluate(
    question="...",
    answer="...",
    context="...",
    expected_answer="...",
)

print(f"Faithfulness Score: {result.faithfulness}")
```

---

## 평가 시스템

### 평가 지표

| 지표 | 목표 | 설명 |
|------|------|------|
| QnA Hit Rate | > 60% | QnA DB에서 바로 답변되는 비율 |
| Faithfulness | > 0.9 | 약관 RAG 답변의 사실 기반 정도 |
| Answer Relevance | > 0.85 | 질문과 답변의 관련성 |
| Similarity | > 0.75 | 생성 답변과 정답 간 의미 유사도 |
| BLEU | > 0.3 | 생성 답변과 정답 간 BLEU 점수 |

### LLM-as-a-Judge 평가

Frontier 모델(Claude, GPT, Gemini)을 사용하여 생성된 답변의 품질을 자동 평가합니다.

**평가 기준 (1-5점):**

| 기준 | 설명 |
|------|------|
| **Correctness** | 정답과의 사실적 일치도 |
| **Helpfulness** | 사용자 질문에 대한 유용성 |
| **Faithfulness** | 제공된 컨텍스트에 대한 충실도 |
| **Fluency** | 자연스러운 한국어 표현 |

### 평가 데이터셋 생성

Frontier 모델을 사용하여 "골든 답변"(정답)이 포함된 평가 데이터셋을 생성할 수 있습니다.

```bash
# QnA Store에서 샘플링하여 데이터셋 생성
python scripts/generate_dataset.py \
    --from-qna \
    --n-samples 50 \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --output data/evaluation/golden_dataset.json

# 질문 파일에서 데이터셋 생성
python scripts/generate_dataset.py \
    --questions data/questions.json \
    --provider openai \
    --model gpt-4o \
    --output data/evaluation/golden_dataset.json

# 드라이런 (API 호출 없이 확인)
python scripts/generate_dataset.py --from-qna --dry-run
```

**데이터셋 생성 옵션:**

| 옵션 | 설명 |
|------|------|
| `--from-qna` | QnA Vector Store에서 샘플링 |
| `--questions <file>` | 질문 JSON 파일에서 로드 |
| `--provider` | `openai`, `anthropic`, `google` 중 선택 |
| `--model` | 사용할 모델 (기본값: gpt-4o / claude-sonnet-4-20250514) |
| `--n-samples` | 샘플 개수 (--from-qna 사용 시) |
| `--categories` | 카테고리 필터 (쉼표 구분) |
| `--seed` | 랜덤 시드 (재현성 확보) |

### 평가 실행

```bash
# 기본 평가 (Similarity, BLEU, Faithfulness)
python scripts/run_evaluation.py \
    --dataset data/evaluation/golden_dataset.json \
    --models "llama3.1:8b" \
    --provider ollama

# LLM-as-a-Judge 평가 활성화
python scripts/run_evaluation.py \
    --dataset data/evaluation/golden_dataset.json \
    --models "llama3.1:8b,mistral:7b" \
    --use-llm-judge \
    --judge-provider openai \
    --judge-model gpt-4o \
    --report

# 테스트 케이스 수 제한
python scripts/run_evaluation.py \
    --dataset data/evaluation/golden_dataset.json \
    --models "llama3.1:8b" \
    --limit 10 \
    --use-llm-judge
```

**평가 CLI 옵션:**

| 옵션 | 설명 |
|------|------|
| `--models` | 평가할 모델 (쉼표 구분) |
| `--dataset` | 평가 데이터셋 JSON 경로 |
| `--provider` | LLM 프로바이더 (vllm, sglang, ollama, openai) |
| `--use-llm-judge` | LLM-as-a-Judge 평가 활성화 |
| `--judge-provider` | Judge 모델 프로바이더 (openai, anthropic, google) |
| `--judge-model` | Judge 모델 (기본값: gpt-4o) |
| `--limit` | 평가 케이스 수 제한 |
| `--report` | Markdown/CSV 리포트 생성 |
| `--no-pipeline` | 파이프라인 없이 예상 답변만 비교 |

### 평가용 의존성 설치

```bash
# 전체 설치 (evaluation 포함)
uv pip install -e ".[all]"

# 또는 evaluation만 설치
uv pip install -e ".[evaluation]"
```

**필요한 API 키:**

| 프로바이더 | 환경변수 |
|-----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |

### 평가 아키텍처

```
평가 데이터셋 생성:
┌─────────────────────────────────────────────────────────────┐
│  questions.json / QnA Store                                  │
│            │                                                 │
│            ▼                                                 │
│  ┌─────────────────────────┐                                 │
│  │  Frontier Client        │  (Claude/GPT/Gemini)           │
│  │  - Golden Answer 생성    │                                 │
│  └─────────────────────────┘                                 │
│            │                                                 │
│            ▼                                                 │
│  golden_dataset.json (질문 + 골든 답변)                       │
└─────────────────────────────────────────────────────────────┘

평가 실행:
┌─────────────────────────────────────────────────────────────┐
│  golden_dataset.json                                         │
│            │                                                 │
│            ▼                                                 │
│  ┌─────────────────────────┐                                 │
│  │  RAG Pipeline           │  (평가 대상 모델)               │
│  │  - 답변 생성             │                                 │
│  └─────────────────────────┘                                 │
│            │                                                 │
│            ▼                                                 │
│  ┌─────────────────────────┐                                 │
│  │  LLM Evaluator          │                                 │
│  │  - Similarity, BLEU     │  (기본 메트릭)                   │
│  │  - Faithfulness         │                                 │
│  └─────────────────────────┘                                 │
│            │                                                 │
│            ▼                                                 │
│  ┌─────────────────────────┐                                 │
│  │  LLM-as-a-Judge         │  (선택적)                       │
│  │  - Correctness          │                                 │
│  │  - Helpfulness          │                                 │
│  │  - Faithfulness         │                                 │
│  │  - Fluency              │                                 │
│  └─────────────────────────┘                                 │
│            │                                                 │
│            ▼                                                 │
│  results/eval_<timestamp>.json + report.md                   │
└─────────────────────────────────────────────────────────────┘
```

### 평가 모듈 구조

```
src/evaluation/
├── __init__.py           # 패키지 exports
├── evaluator.py          # 기본 메트릭 (Similarity, BLEU, Faithfulness)
├── runner.py             # 배치 평가 실행기
├── report.py             # Markdown/CSV 리포트 생성
├── frontier_client.py    # Claude/GPT/Gemini 통합 클라이언트
├── judge_prompts.py      # LLM-as-a-Judge 프롬프트
├── llm_judge.py          # LLM-as-a-Judge 구현
└── dataset_generator.py  # 골든 답변 데이터셋 생성

scripts/
├── run_evaluation.py     # 평가 실행 CLI
└── generate_dataset.py   # 데이터셋 생성 CLI
```

---

## License

MIT License
