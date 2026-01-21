# Retrieval System

본 시스템은 QnA(FAQ)와 ToS(약관) 두 종류의 데이터에 대해 각각 최적화된 retrieval 방식을 제공합니다.

## 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Retrieval System                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐       ┌─────────────────────────────────┐  │
│  │   QnA Retrieval     │       │       ToS Retrieval             │  │
│  │   (단순 벡터 검색)   │       │    (Hybrid Search 3-way)        │  │
│  │                     │       │                                 │  │
│  │  ┌───────────────┐  │       │  ┌─────────┐  ┌─────────────┐  │  │
│  │  │ Vector Search │  │       │  │ Vector  │  │ Rule-based  │  │  │
│  │  │  (Question    │  │       │  │ Search  │  │   Matcher   │  │  │
│  │  │   Embedding)  │  │       │  │  (50%)  │  │    (30%)    │  │  │
│  │  └───────────────┘  │       │  └─────────┘  └─────────────┘  │  │
│  │                     │       │       ┌─────────────┐          │  │
│  └─────────────────────┘       │       │   Triplet   │          │  │
│                                │       │    Store    │          │  │
│                                │       │    (20%)    │          │  │
│                                │       └─────────────┘          │  │
│                                └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## QnA Retrieval

FAQ 데이터를 위한 단순하고 빠른 벡터 기반 검색입니다.

### 작동 방식

1. **Question Embedding**: 질문 텍스트를 E5 모델로 임베딩
2. **Cosine Similarity**: ChromaDB에서 유사도 기반 검색
3. **Answer 반환**: 가장 유사한 질문의 답변을 반환

### 특징

- **검색 대상**: 질문(Question)을 임베딩하여 저장
- **반환 데이터**: 유사 질문과 해당 답변(Answer)
- **임베딩 모델**: E5 (query/document prefix 지원)

### 코드 예시

```python
from src.vectorstore import QnAVectorStore

qna_store = QnAVectorStore()

# 검색
results = qna_store.search(
    query="환불은 어떻게 하나요?",
    n_results=5,
    category_filter="결제",       # 선택적 카테고리 필터
    score_threshold=0.7,          # 최소 유사도 점수
)

for r in results:
    print(f"Q: {r.question}")
    print(f"A: {r.answer}")
    print(f"Score: {r.score}")
```

### 메타데이터 스키마

| 필드 | 설명 |
|------|------|
| question | FAQ 질문 텍스트 |
| answer | FAQ 답변 텍스트 |
| category | 질문 카테고리 |
| sub_category | 하위 카테고리 |
| source | 출처 ("FAQ" 또는 "human_agent") |
| source_url | 원본 URL |
| human_verified | 사람 검증 여부 |

---

## ToS Retrieval (Hybrid Search)

약관 문서는 단순 벡터 검색만으로는 정확도가 낮아, **3가지 검색 방식을 결합한 Hybrid Search**를 사용합니다.

### 왜 Hybrid Search가 필요한가?

약관 문서의 특성:
- "제3조"와 같은 **정확한 조항 참조**가 빈번함
- "환불", "위약금" 등 **법률 용어**의 정확한 매칭 필요
- "회사는 X를 할 수 있다"와 같은 **관계 구조** 이해 필요

단순 벡터 검색은 의미적 유사성은 찾지만, 이러한 구조적/규칙적 패턴을 놓칠 수 있습니다.

### 3-Way Hybrid Search

```
최종 점수 = α × Vector Score + β × Rule Score + γ × Triplet Score

기본 가중치: α=0.5, β=0.3, γ=0.2
```

#### 1. Vector Search (50%)

의미적 유사성 기반 검색:
- E5 임베딩 모델 사용
- 청크 단위로 약관 내용 임베딩
- Cosine similarity로 유사도 계산

#### 2. Rule-based Matcher (30%)

패턴 매칭 기반 점수 부스팅:

**a) 조항 참조 매칭**
```
쿼리: "제3조 2항 알려줘"
     ↓ 정규식 추출
SectionRef(article_num=3, clause_num=2)
     ↓ 문서 매칭
"제3조 (이용조건)" → 매칭 성공 → +1.0 boost
```

**b) 키워드 가중치**

법률/약관 특화 키워드에 가중치 부여:

| 카테고리 | 키워드 | 가중치 |
|----------|--------|--------|
| 금전 관련 | 환불, 위약금 | 0.9 |
| 금전 관련 | 수수료, 손해배상 | 0.8 |
| 계약 관련 | 해지, 해제 | 0.8 |
| 계약 관련 | 취소, 철회 | 0.7 |
| 책임 관련 | 면책 | 0.9 |
| 책임 관련 | 책임 | 0.7 |
| 개인정보 | 개인정보 | 0.8 |

#### 3. Triplet Store (20%)

**Subject-Predicate-Object** 관계 기반 검색:

**추출 패턴:**
```
"회사는 서비스를 거부할 수 있다"
    ↓
Triplet(subject="회사", predicate="가능", object="서비스 거부")

"고객은 환불을 요청해야 한다"
    ↓
Triplet(subject="고객", predicate="의무", object="환불 요청")

"약관이라 함은 계약 조건을 말한다"
    ↓
Triplet(subject="약관", predicate="정의", object="계약 조건")
```

**지원 패턴:**
| 패턴 | 예시 | 추출 결과 |
|------|------|-----------|
| 가능 | "회사는 X를 할 수 있다" | (회사, 가능, X) |
| 의무 | "고객은 X를 해야 한다" | (고객, 의무, X) |
| 정의 | "X라 함은 Y를 말한다" | (X, 정의, Y) |
| 금지 | "X하여서는 아니 된다" | (행위자, 금지, X) |
| 면책 | "회사는 X에 대해 책임지지 않는다" | (회사, 면책, X) |

**검색 방식:**
- subject, predicate, object 각각에 대해 fuzzy matching
- 매칭된 triplet의 source_chunk_id로 원본 청크 연결
- 점수 정규화 후 가중치 적용

### 사용 예시

```python
from src.tos_search import ToSHybridSearch, HybridSearchConfig
from src.vectorstore import ToSVectorStore

# 벡터 스토어 초기화
tos_store = ToSVectorStore()

# Hybrid Search 초기화
hybrid_search = ToSHybridSearch(
    vector_store=tos_store,
    config=HybridSearchConfig(
        vector_weight=0.5,
        rule_weight=0.3,
        triplet_weight=0.2,
    ),
)

# Triplet 인덱스 빌드 (최초 1회)
hybrid_search.build_triplet_index()

# 검색
results = hybrid_search.search(
    query="제5조 환불 규정",
    n_results=5,
)

for r in results:
    print(f"Title: {r.section_title}")
    print(f"Combined Score: {r.combined_score:.2f}")
    print(f"  - Vector: {r.vector_score:.2f}")
    print(f"  - Rule: {r.rule_score:.2f}")
    print(f"  - Triplet: {r.triplet_score:.2f}")
    
    # 검색 설명
    explanation = hybrid_search.get_search_explanation(r)
    print(f"Explanation: {explanation}")
```

### HybridSearchResult 구조

```python
@dataclass
class HybridSearchResult:
    chunk_id: str
    section_title: str
    section_content: str
    document_title: str
    category: str
    effective_date: str
    source_url: str
    
    # 점수
    combined_score: float      # 최종 결합 점수
    vector_score: float        # 벡터 유사도 점수
    rule_score: float          # 규칙 기반 점수
    triplet_score: float       # Triplet 매칭 점수
    
    # 매칭 정보
    matched_keywords: list[str]      # 매칭된 키워드
    section_ref_match: bool          # 조항 참조 매칭 여부
    matched_triplets: list[dict]     # 매칭된 triplets
```

---

## 요약 비교

| 특성 | QnA Retrieval | ToS Retrieval |
|------|---------------|---------------|
| 검색 방식 | Vector Only | Hybrid (Vector + Rule + Triplet) |
| 임베딩 대상 | Question | Section Content |
| 조항 참조 지원 | X | O (Rule-based) |
| 키워드 부스팅 | X | O (30개 법률 용어) |
| 관계 추출 | X | O (Triplet Store) |
| 적합한 데이터 | FAQ, 짧은 Q&A | 약관, 법률 문서, 계약서 |
