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
| **LLM** | OpenAI GPT-4o / GPT-4o-mini |
| **Embedding** | Qwen3-Embedding (0.6B/8B), multilingual-e5-large |
| **Vector DB** | ChromaDB |
| **MCP** | FastMCP |
| **Package Manager** | uv |

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
    │ (검색 실패)
    ▼
┌─────────────────────────────────────────────────────┐
│  [STEP 3] No Context 응답                            │
│  • 관련 정보 없음 안내                                │
│  • 고객센터 연결 안내                                 │
└─────────────────────────────────────────────────────┘
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
│   └── vectordb/               # ChromaDB 저장소
│       ├── qna/
│       └── tos/
├── src/
│   ├── crawlers/               # 데이터 크롤러
│   ├── vectorstore/            # Vector DB (ChromaDB)
│   │   ├── qna_store.py
│   │   ├── tos_store.py
│   │   └── embeddings.py
│   ├── tos_search/             # ToS 하이브리드 검색
│   │   ├── rule_matcher.py     # 조항 패턴 + 키워드 매칭
│   │   ├── triplet_store.py    # Subject-Predicate-Object 검색
│   │   └── hybrid_search.py    # Vector + Rule + Triplet 결합
│   ├── llm/                    # LLM 클라이언트
│   │   └── openai_client.py
│   ├── pipeline/               # RAG 파이프라인
│   │   └── rag_pipeline.py
│   ├── mcp/                    # MCP 서버
│   │   └── server.py
│   └── verifier/               # Hallucination 검증
├── scripts/
│   ├── ingest_qna.py           # QnA 데이터 적재
│   ├── ingest_tos.py           # ToS 데이터 적재
│   ├── run_pipeline.py         # CLI 실행
│   └── run_mcp_server.py       # MCP 서버 실행
├── configs/
│   ├── embedding_config.yaml
│   └── claude_desktop_config.example.json
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
```

### 3. 파이프라인 실행

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY="sk-..."

# Interactive 모드
python scripts/run_pipeline.py

# 단일 질문
python scripts/run_pipeline.py -q "계좌 해지 방법이 뭐야?"

# 검색만 테스트
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
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

---

## 설정

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

## 평가 지표

| 지표 | 목표 | 설명 |
|------|------|------|
| QnA Hit Rate | > 60% | QnA DB에서 바로 답변되는 비율 |
| Faithfulness | > 0.9 | 약관 RAG 답변의 사실 기반 정도 |
| Answer Relevance | > 0.85 | 질문과 답변의 관련성 |

---

## License

MIT License
