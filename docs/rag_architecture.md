# RAG 아키텍처

`RAGPipeline`은 외부 인터페이스를 제공하는 얇은 퍼사드(facade)이며, 실제 오케스트레이션은 `src/graph/` 아래의 LangGraph 상태 머신이 담당합니다.

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `src/pipeline/rag_pipeline.py` | 스토어, LLM, 검증기, 임계값 초기화 및 그래프 실행 |
| `src/pipeline/models.py` | `PipelineResponse`, `ResponseSource` 공유 타입 |
| `src/graph/graph.py` | 상태 그래프 빌드 및 컴파일 |
| `src/graph/state.py` | `GraphState` 데이터 클래스 |
| `src/graph/nodes/` | search, generate, verify, format 노드 구현 |
| `src/graph/edges/routers.py` | 임계값 기반 라우팅 함수 |

---

## 전체 파이프라인 흐름

```
User Query
    │
    ▼
[search_qna]              ← QnA 벡터 DB 의미론적 검색
    │
    ├─ score ≥ 0.80 ──► [generate_qna_answer]    → FAQ 정답 생성
    ├─ score ≥ 0.70 ──► [generate_qna_limited]   → FAQ 부분 답변 생성
    └─ score < 0.70 ──► [search_tos]             ← ToS 벡터/하이브리드 검색
                              │
                              ├─ score ≥ 0.65 ──► [generate_tos_answer]    → 약관 근거 정답 생성
                              ├─ score ≥ 0.55 ──► [generate_tos_limited]   → 약관 부분 답변 생성
                              ├─ score ≥ 0.40 ──► [generate_clarification] → 추가 질문 유도
                              └─ score < 0.40 ──► [generate_no_context]    → 상담 채널 안내
                                        │
    ┌─────────────────────────────────┘
    │   (모든 생성 노드 → verify_answer)
    ▼
[verify_answer]           ← 환각 검증 (3-Layer Defense)
    │
    ▼
[format_response]         ← 최종 응답 포맷팅
    │
    ▼
PipelineResponse
```

---

## 라우팅 임계값

### QnA 라우터 (`route_qna`)

| 조건 | 다음 노드 | 동작 |
|------|-----------|------|
| `qna_score ≥ 0.80` | `generate_qna_answer` | FAQ에서 정확한 답변 생성 |
| `0.70 ≤ qna_score < 0.80` | `generate_qna_limited` | 유사 FAQ 기반 제한적 답변; ToS 폴백 없음 |
| `qna_score < 0.70` | `search_tos` | ToS 문서 검색으로 폴백 |

### ToS 라우터 (`route_tos`)

| 조건 | 다음 노드 | 동작 |
|------|-----------|------|
| `tos_score ≥ 0.65` | `generate_tos_answer` | 약관 조항에 근거한 정확한 답변 생성 |
| `0.55 ≤ tos_score < 0.65` | `generate_tos_limited` | 관련 조항 기반 제한적 답변 생성 |
| `0.40 ≤ tos_score < 0.55` | `generate_clarification` | 더 구체적인 질문 요청 |
| `tos_score < 0.40` | `generate_no_context` | 관련 정보 없음, 상담 채널 안내 |

### 전체 임계값 기본값

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `qna_threshold` | **0.80** | QnA 고신뢰 임계값 |
| `qna_mid_threshold` | **0.70** | QnA 중간 신뢰 임계값 |
| `tos_threshold` | **0.65** | ToS 고신뢰 임계값 |
| `tos_mid_threshold` | **0.55** | ToS 중간 신뢰 임계값 |
| `tos_low_threshold` | **0.40** | ToS 낮은 신뢰 임계값 |
| `verification_threshold` | **0.70** | 검증 통과 최소 신뢰도 |

`RAGPipeline` 생성자 또는 `configs/agent_config.yaml`에서 조정할 수 있습니다.

---

## 단계별 상세 설명

### 1단계: QnA 검색

- **목표**: 사전 정의된 FAQ로 즉시 답변 가능한 질문 처리
- **방식**: QnA Chroma 컬렉션에 대한 밀집 벡터 검색
- **결과 정규화**: `question`, `answer`, `category`, `sub_category`, `score`, `source`, `source_url` 포함 딕셔너리로 변환

### 2단계: ToS 검색

- **목표**: FAQ로 해결되지 않는 정책·약관 기반 질문 처리
- **방식**: 기본은 벡터 검색, `enable_hybrid_tos_search=True` 시 하이브리드 검색으로 전환
- **섹션 참조 추출**: `제1조` 같은 명시적 조항 번호는 쿼리에서 파싱하여 `state.section_reference`에 저장
- **하이브리드 검색 활성화 시**: `final_score`, `combined_score`, 리랭크 출력, 매칭 키워드, 매칭 트리플릿이 `state.metadata`에 기록

> 자세한 하이브리드 검색 메커니즘은 [검색 및 랭킹](search_ranking.md)을 참고하세요.

### 3단계: 답변 검증 (`verify_answer`)

- **목표**: 생성된 답변의 환각(hallucination) 방지
- **실행 조건**: 검증이 활성화되어 있고 응답 소스가 ToS이며 `response_mode`가 `answer` 또는 `limited_answer`인 경우에만 실행; 나머지(QnA 답변, 명확화 요청, 핸드오프)는 no-op
- **출력 필드**: `verified`, `verification_score`, `verification_issues`, `metadata["verification_reasoning"]`

3단계 방어 체계:

```
Layer 1 — Citation Check
    [참조: 제N조] 인용이 실제 검색된 컨텍스트에 존재하는지 확인

Layer 2 — Rule-based Check
    "아마도", "추측컨대", "일반적으로" 등 불확실 표현 패턴 탐지

Layer 3 — LLM-based Check
    보조 LLM 호출로 답변이 컨텍스트에 충실한지 심층 검증
    → { "verified": bool, "confidence": float, "issues": [...] }
```

신뢰도 계산:
- 인용 오류: `-0.30`
- 불확실 표현 1건당: `-0.15`
- LLM 검증 결과: 현재 점수와 평균 (가중치 50%)
- 최종: `verified = (confidence ≥ threshold) AND (issues가 없음)`

---

## 공개 API 요약

```python
pipeline = RAGPipeline(
    enable_verification=True,
    enable_hybrid_tos_search=False,
)

# 전체 파이프라인 실행
response: PipelineResponse = pipeline.query("계좌 해지 방법이 뭐야?")

# 직접 검색 (LLM 불필요)
qna_results = pipeline.search_qna("비밀번호", n_results=5)
tos_results = pipeline.search_tos("제1조", top_k=3)  # n_results 별칭
```

`PipelineResponse` 주요 필드:

| 필드 | 타입 | 설명 |
|------|------|------|
| `answer` | `str` | 생성된 답변 텍스트 |
| `source` | `ResponseSource` | `QNA` / `TOS` / `NO_CONTEXT` |
| `confidence` | `float` | 검색 단계의 유사도 점수 |
| `response_mode` | `str` | `answer` / `limited_answer` / `clarification` / `no_context` |
| `verified` | `bool` | 검증 통과 여부 |
| `verification_score` | `float` | 검증 신뢰도 (0–1) |
| `citations` | `list[str]` | 답변에서 추출된 인용 목록 |
