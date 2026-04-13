# 검색 및 랭킹

복잡한 약관 문서(ToS)에서 정확한 조항을 찾기 위해 다단계 검색 전략을 사용합니다. 기본은 벡터 검색이며, `enable_hybrid_tos_search=True`로 하이브리드 검색과 리랭킹을 활성화할 수 있습니다.

---

## 1. 하이브리드 검색

구현: `src/tos_search/hybrid_search.py`

세 가지 신호를 가중합산하여 최종 점수를 계산합니다.

```
CombinedScore = α × VectorScore + β × RuleScore + γ × TripletScore
                (α=0.5)          (β=0.3)          (γ=0.2)
```

### 1-1. 벡터 검색 (Dense / Semantic)

- `multilingual-e5-large` 임베딩을 통한 의미론적 유사도 검색
- "환불 요청" 쿼리가 "반환 규정" 조항과 매칭되는 등 표면적 키워드가 달라도 의미 기반으로 검색 가능
- ChromaDB 코사인 유사도로 후보를 빠르게 추출

### 1-2. 규칙/키워드 매처 (Rule-based / Lexical)

구현: `src/tos_search/rule_matcher.py`

- **조항 번호 추출**: `제1조`, `제5조 제2항` 같은 명시적 참조를 정규식으로 파싱
- **키워드 매칭**: 쿼리 내 핵심 단어가 문서 제목·헤더에 존재하는지 확인
- 벡터 검색이 놓칠 수 있는 **정밀한 법률 용어**와 **조항 번호** 검색에 강점

### 1-3. 트리플릿 매처 (Knowledge Graph)

구현: `src/tos_search/triplet_store.py`

- ToS 청크에서 추출한 `(주어, 관계, 목적어)` 트리플릿 인덱스를 활용
- 쿼리와 트리플릿 간 관계 수준 매칭 점수를 계산
- 청크별 트리플릿 점수 정규화:

```
TripletScore_i = Σ(매칭 트리플릿 점수_i) / max_j(Σ 매칭 트리플릿 점수_j)
```

- 트리플릿 인덱스가 비어 있거나 매칭이 없으면 `triplet_score = 0.0`

### 가중치 설정 (`HybridSearchConfig`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `vector_weight` (α) | `0.5` | 벡터 검색 가중치 |
| `rule_weight` (β) | `0.3` | 규칙 매처 가중치 |
| `triplet_weight` (γ) | `0.2` | 트리플릿 매처 가중치 |

---

## 2. Cross-Encoder 리랭킹

하이브리드 검색으로 후보군(예: Top 20)을 추린 후, Cross-Encoder로 정밀 재순위화합니다.

### Bi-Encoder vs. Cross-Encoder 비교

| 구분 | Bi-Encoder | Cross-Encoder |
|------|-----------|---------------|
| **용도** | 빠른 후보 검색 | 정밀 재순위화 |
| **입력** | 쿼리, 문서 각각 별도 인코딩 | `(쿼리, 문서)` 쌍을 함께 처리 |
| **속도** | 빠름 | 느림 |
| **정확도** | 보통 | 높음 |

Cross-Encoder는 쿼리와 문서를 함께 읽기 때문에 문맥 내 미묘한 관련성까지 포착합니다.

### 처리 과정

```
1. 하이브리드 검색으로 확장된 후보 추출
   (n_results × 3, 최소 rerank_candidates 수 이상)

2. Cross-Encoder에 (쿼리, 문서 제목 + 본문) 쌍 입력

3. 리랭크 점수 Min-Max 정규화 (후보 내)

4. 최종 점수 계산:
   FinalScore = (1 - w) × CombinedScore + w × NormalizedRerankScore
                                          (w = rerank_weight, 기본 0.3)

5. FinalScore 기준 정렬 후 상위 결과 반환
```

> 리랭크 점수가 없거나 유효하지 않을 경우 `FinalScore = CombinedScore` 로 폴백됩니다.

### 리랭커 설정 (`configs/embedding_config.yaml`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `rerank_enabled` | `False` | 리랭킹 활성화 여부 |
| `rerank_model` | `BAAI/bge-reranker-v2-m3` | Cross-Encoder 모델 |
| `rerank_candidates` | `20` | 리랭킹 입력 후보 수 |
| `rerank_weight` (w) | `0.3` | 리랭크 점수 반영 비율 |
| `rerank_batch_size` | `32` | 배치 크기 |

---

## 검색 모드 요약

| 모드 | 활성화 조건 | 사용 신호 |
|------|------------|-----------|
| 기본 벡터 검색 | 항상 | 벡터 유사도만 |
| 하이브리드 검색 | `enable_hybrid_tos_search=True` | 벡터 + 규칙 + 트리플릿 |
| 하이브리드 + 리랭킹 | 하이브리드 + `rerank_enabled=True` | 위 모두 + Cross-Encoder |
