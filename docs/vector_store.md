# 벡터 검색 시스템

QnA와 ToS 데이터를 저장·검색하는 벡터 DB 레이어입니다. **ChromaDB**를 백엔드로 사용하며, 한국어 특화 임베딩 모델로 의미론적 유사도를 계산합니다.

---

## ChromaDB 구성

| 항목 | 값 |
|------|----|
| 저장 방식 | 로컬 파일 시스템 (`data/vectordb/`) |
| 거리 메트릭 | 코사인 유사도 (`hnsw:space: cosine`) |
| QnA 컬렉션 경로 | `data/vectordb/qna` |
| ToS 컬렉션 경로 | `data/vectordb/tos` |

---

## 임베딩 모델 (Bi-Encoder)

Bi-Encoder는 쿼리와 문서를 **각각** 고정 크기 벡터로 변환하여 코사인 유사도로 비교합니다. 빠른 후보 검색에 적합합니다.

### 기본 모델: `intfloat/multilingual-e5-large`

- **벡터 차원**: 1024
- **특징**: 한국어를 포함한 다국어 검색 태스크에서 표준 BERT 대비 우수한 성능
- **입력 접두사 규칙** (필수):
  - 쿼리: `query: 환불 규정이 어떻게 되나요?`
  - 문서: `passage: 제1조 본 약관은...`

### 지원 모델 목록 (`configs/embedding_config.yaml`)

| 모델 | 차원 | 특징 |
|------|------|------|
| `intfloat/multilingual-e5-large` | 1024 | 기본값, 다국어 고성능 |
| `intfloat/multilingual-e5-base` | 768 | 빠름, 낮은 메모리 |
| `BAAI/bge-m3` | 1024 | 다국어 + 긴 컨텍스트 지원 |
| `text-embedding-3-small` | 1536 | OpenAI (API 키 필요) |

---

## 데이터 스키마

### QnA 컬렉션

벡터화 대상은 `question` 텍스트입니다. 나머지 필드는 메타데이터로 저장됩니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `question` | `str` | 질문 텍스트 (벡터화) |
| `answer` | `str` | 답변 텍스트 (메타데이터) |
| `category` | `str` | 카테고리 (예: "결제", "계좌") |
| `sub_category` | `str` | 세부 카테고리 |
| `source` | `str` | 출처 ("FAQ", "Manual" 등) |
| `source_url` | `str` | 원본 URL |

### ToS 컬렉션

벡터화 대상은 `section_content` 텍스트입니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `document_title` | `str` | 약관 문서명 (예: "서비스이용약관") |
| `section_title` | `str` | 조항 제목 (예: "제1조 (목적)") |
| `section_content` | `str` | 조항 본문 (벡터화) |
| `category` | `str` | 약관 카테고리 |
| `chunk_index` | `int` | 문서 내 청크 순서 |

---

## 주요 클래스

### `QnAVectorStore`

```python
from src.vectorstore import QnAVectorStore

store = QnAVectorStore(persist_directory="data/vectordb/qna")

# 검색
results = store.search("비밀번호 변경", n_results=5)
# results: list[QnASearchResult]  (question, answer, category, score, ...)

# 데이터 수
store.count()
```

### `ToSVectorStore`

```python
from src.vectorstore import ToSVectorStore

store = ToSVectorStore(
    persist_directory="data/vectordb/tos",
    enable_hybrid_search=True,   # 하이브리드 검색 활성화
)

results = store.search("해지 절차", n_results=5)
# results: list[ToSSearchResult]  (document_title, section_title, section_content, score, ...)
```

> `enable_hybrid_search=True`일 때는 내부적으로 `HybridSearch`를 통해 벡터 + 룰 + 트리플릿 점수를 결합합니다. 자세한 내용은 [검색 및 랭킹](search_ranking.md)을 참고하세요.
