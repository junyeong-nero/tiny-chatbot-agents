# Tiny Chatbot Agents

**한국어 고객 서비스 챗봇 — 이용약관(ToS) 및 자주 묻는 질문(QnA) 기반 RAG 파이프라인**

LangGraph로 오케스트레이션되는 계층적 검색-증강 생성(RAG) 시스템입니다. 사전 정의된 QnA 데이터베이스를 우선 탐색하고, 매칭 신뢰도가 낮을 때 이용약관(ToS) 문서로 폴백하며, 최종 답변을 생성하기 전 환각(hallucination) 여부를 자동으로 검증합니다.

---

## 목차

- [핵심 기능](#-핵심-기능)
- [아키텍처](#-아키텍처)
  - [전체 파이프라인 흐름](#전체-파이프라인-흐름)
  - [라우팅 임계값 상세](#라우팅-임계값-상세)
  - [ToS 하이브리드 검색](#tos-하이브리드-검색)
  - [환각 검증 (3-Layer Defense)](#환각-검증-3-layer-defense)
- [프로젝트 구조](#-프로젝트-구조)
- [시작하기](#-시작하기)
- [사용법](#-사용법)
- [설정](#-설정)
- [기술 스택](#-기술-스택)

---

## ✨ 핵심 기능

| 기능 | 설명 |
|------|------|
| **Graph-Orchestrated RAG** | LangGraph 상태 머신으로 검색 → 생성 → 검증 → 포맷 노드를 구성 |
| **계층적 이중 검색** | 정제된 QnA 매칭 우선, 신뢰도 낮을 시 ToS 문서로 자동 폴백 |
| **ToS 하이브리드 검색** | 벡터 검색 + 규칙 기반 매칭 + Knowledge Graph 트리플릿 결합 |
| **Advanced Reranking** | Bi-Encoder(E5)로 빠른 후보 검색, Cross-Encoder(BGE-Reranker)로 정밀 재순위화 |
| **환각 검증** | 인용 검사 → 패턴 검사 → LLM 검사, 3단계 방어 시스템 |
| **MCP 서버** | Claude Desktop 등 MCP 클라이언트와 통합 가능 |
| **로컬 LLM 지원** | vLLM, Ollama 등 OpenAI 호환 로컬 추론 서버 지원 |
| **LLM-as-a-Judge 평가** | 한국어 특화 메트릭, 병렬 실행 평가 프레임워크 |

---

## 🏗️ 아키텍처

### 전체 파이프라인 흐름

사용자 질의는 다음 순서로 처리됩니다.

```
User Query
    │
    ▼
[1] search_qna          ← QnA 벡터 DB 검색
    │
    ├─ score ≥ 0.80 ──► generate_qna_answer    (FAQ 정답)
    ├─ score ≥ 0.70 ──► generate_qna_limited   (FAQ 부분 답변)
    └─ score < 0.70 ──► [2] search_tos
                              │
                              ├─ score ≥ 0.65 ──► generate_tos_answer    (약관 기반 정답)
                              ├─ score ≥ 0.55 ──► generate_tos_limited   (약관 부분 답변)
                              ├─ score ≥ 0.40 ──► generate_clarification (질문 명확화 요청)
                              └─ score < 0.40 ──► generate_no_context    (답변 불가, 상담 유도)
                                        │
    ┌─────────────────────────────────┘
    ▼
[3] verify_answer        ← 환각 검증 (3-Layer Defense)
    │
    ▼
[4] format_response      ← 최종 응답 포맷팅
    │
    ▼
Final Response
```

각 경로의 `source` 값은 `ResponseSource.QNA`, `ResponseSource.TOS`, `ResponseSource.NO_CONTEXT` 중 하나로 반환됩니다.

---

### 라우팅 임계값 상세

#### QnA 라우터

| 조건 | 다음 노드 | 동작 |
|------|-----------|------|
| `qna_score ≥ 0.80` | `generate_qna_answer` | FAQ 데이터에서 정확한 답변 생성 |
| `0.70 ≤ qna_score < 0.80` | `generate_qna_limited` | 유사 FAQ를 바탕으로 제한적 답변 생성 |
| `qna_score < 0.70` | `search_tos` | ToS 문서 검색으로 폴백 |

#### ToS 라우터

| 조건 | 다음 노드 | 동작 |
|------|-----------|------|
| `tos_score ≥ 0.65` | `generate_tos_answer` | 약관 근거로 정확한 답변 생성 |
| `0.55 ≤ tos_score < 0.65` | `generate_tos_limited` | 관련 조항을 바탕으로 제한적 답변 생성 |
| `0.40 ≤ tos_score < 0.55` | `generate_clarification` | 더 구체적인 질문을 유도 |
| `tos_score < 0.40` | `generate_no_context` | 관련 정보 없음, 상담 채널 안내 |

> 임계값은 `RAGPipeline` 생성자 또는 `configs/agent_config.yaml`에서 조정할 수 있습니다.

---

### ToS 하이브리드 검색

`enable_hybrid_tos_search=True`로 활성화하면 세 가지 신호를 결합합니다.

```
최종 점수 = α × vector_score + β × rule_score + γ × triplet_score
           (α=0.5)           (β=0.3)           (γ=0.2)
```

| 구성 요소 | 설명 |
|-----------|------|
| **Vector Search** | `multilingual-e5-large` 임베딩 기반 의미론적 유사도 |
| **Rule Matcher** | 조 번호(제N조), 키워드 등 정규식 기반 정밀 매칭 |
| **Triplet Store** | Knowledge Graph에서 추출한 (주어, 관계, 목적어) 트리플릿 매칭 |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3`로 최종 후보 재순위화 (선택적) |

---

### 환각 검증 (3-Layer Defense)

`verify_answer` 노드는 생성된 답변이 컨텍스트에 근거하는지 3단계로 검증합니다.

```
Layer 1: Citation Check
    └─ [참조: 제N조] 형식의 인용이 실제 검색 컨텍스트에 존재하는지 확인

Layer 2: Rule-based Check
    └─ "아마도", "추측컨대", "일반적으로" 등 불확실 표현 패턴 탐지

Layer 3: LLM-based Check
    └─ 보조 LLM 호출로 답변이 컨텍스트에 충실한지 검증
         → {"verified": bool, "confidence": float, "issues": [...]}
```

최종 신뢰도 = `(인용 점수 + 패턴 점수 + LLM 신뢰도) / 2`

`verification_score ≥ 0.7`을 만족하고 `issues`가 없는 경우 `verified=True`로 반환됩니다.

> **참고**: 검증 메타데이터(`verified`, `verification_score`)는 ToS `answer` 및 `limited_answer` 응답에만 첨부됩니다.

---

## 📁 프로젝트 구조

```
tiny-chatbot-agents/
├── main.py                    # 통합 CLI 진입점
├── configs/
│   ├── agent_config.yaml      # 임계값, LLM 제공자, 검색 파라미터
│   └── embedding_config.yaml  # 임베딩 모델 및 리랭커 설정
├── data/
│   └── vectordb/              # ChromaDB 데이터 (qna/, tos/)
├── src/
│   ├── graph/                 # LangGraph 핵심 모듈
│   │   ├── graph.py           # 그래프 빌드 및 컴파일
│   │   ├── state.py           # GraphState 데이터 클래스
│   │   ├── nodes/             # 각 노드 구현 (search, generate, verify, format)
│   │   └── edges/             # 라우팅 로직 (route_qna, route_tos)
│   ├── pipeline/
│   │   ├── rag_pipeline.py    # RAGPipeline 퍼사드 (외부 인터페이스)
│   │   └── models.py          # PipelineResponse, ResponseSource
│   ├── vectorstore/           # ChromaDB 래퍼 (QnAVectorStore, ToSVectorStore)
│   ├── tos_search/            # 하이브리드 검색, 리랭커, 룰 매처, 트리플릿
│   ├── verifier/              # 환각 검증 (AnswerVerifier, VerificationResult)
│   ├── llm/                   # LLM 클라이언트 추상화 (local, openai)
│   ├── mcp/                   # MCP 서버 구현
│   ├── crawlers/              # QnA/ToS 웹 크롤러
│   ├── evaluation/            # LLM-as-a-Judge 평가 프레임워크
│   └── streamlit_app.py       # 웹 UI
└── tests/                     # 단위 테스트 및 회귀 테스트
```

---

## 🚀 시작하기

### 사전 요구 사항

- Python 3.11 이상
- [`uv`](https://github.com/astral-sh/uv) (권장) 또는 pip
- (선택) vLLM, Ollama 등 로컬 LLM 서버 또는 OpenAI API 키

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/tiny-chatbot-agents.git
cd tiny-chatbot-agents

# 2. 의존성 설치 (uv 권장)
uv sync

# pip 사용 시
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 환경 변수 설정

```bash
# LLM 제공자 선택: vllm | sglang | ollama | openai
export LLM_PROVIDER="vllm"
export VLLM_API_BASE="http://localhost:8000/v1"

# OpenAI로 테스트할 경우
export OPENAI_API_KEY="sk-..."
```

---

## 💻 사용법

### 1. CLI 파이프라인

```bash
# 대화형 모드
python main.py pipeline

# 단일 쿼리
python main.py pipeline -q "계좌 해지 방법이 뭐야?"

# LLM 없이 DB 직접 검색
python main.py pipeline --search-qna "비밀번호"
python main.py pipeline --search-tos "제1조"
```

### 2. 웹 인터페이스 (Streamlit)

```bash
python main.py streamlit
# 또는
streamlit run src/streamlit_app.py
```

- RAG 파이프라인과 채팅
- QnA / ToS DB 독립 검색
- LLM 제공자 및 검증 설정 동적 변경

<p align="center">
  <img src="assets/3.png" width="32%">
  <img src="assets/4.png" width="32%">
  <img src="assets/5.png" width="32%">
</p>

### 3. MCP 서버 (Claude Desktop 연동)

```bash
python main.py mcp
```

`claude_desktop_config.json`에 아래를 추가합니다.

```json
{
  "mcpServers": {
    "rag-chatbot": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/tiny-chatbot-agents",
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### 4. 데이터 수집 및 평가

```bash
# 원시 데이터 크롤링
python main.py crawl qna
python main.py crawl tos

# 벡터 DB 적재
python main.py ingest-qna
python main.py ingest-tos

# 평가 실행
python main.py evaluate --models "default" --report
```

---

## ⚙️ 설정

### `configs/agent_config.yaml`

```yaml
qna:
  threshold: 0.85       # QnA 고신뢰 임계값
  n_results: 3

tos:
  threshold: 0.7        # ToS 고신뢰 임계값
  n_results: 5

verifier:
  enabled: true
  confidence_threshold: 0.7
  require_citations: true
  use_llm_verification: true

llm:
  provider: "openai"    # vllm | sglang | ollama | openai
  base_url: "http://localhost:8000/v1"
  model: "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
  temperature: 0.1
```

### `configs/embedding_config.yaml`

임베딩 모델(`multilingual-e5-large`)과 리랭커(`BAAI/bge-reranker-v2-m3`) 설정을 관리합니다.

---

## 🛠️ 기술 스택

| 범주 | 라이브러리 |
|------|-----------|
| **워크플로 오케스트레이션** | `langgraph`, `langchain-core` |
| **벡터 DB** | `ChromaDB` |
| **임베딩 (Bi-Encoder)** | `intfloat/multilingual-e5-large` |
| **리랭킹 (Cross-Encoder)** | `BAAI/bge-reranker-v2-m3` |
| **LLM 인터페이스** | OpenAI 호환 API (vLLM / Ollama) |
| **MCP** | `fastmcp` |
| **웹 UI** | `streamlit` |
| **크롤링** | `playwright` |
