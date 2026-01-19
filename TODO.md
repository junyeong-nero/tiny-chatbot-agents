## 🗺️ 프로젝트 로드맵

> ✅ Phase 1-3 핵심 기능 구현 완료 (2026-01-19)

---

## 남은 TODO

### Phase 2: 검증 평가 도구
- [ ] Ragas 평가 파이프라인 구축
  - `Faithfulness`: 사실 기반 정도
  - `Answer Relevance`: 질문-답변 관련성
  - `Context Precision`: 검색 정확도
- [ ] 평가 결과 대시보드 구현

### Phase 3: 상담원 연결 인터페이스
- [ ] 실시간 상담원 연결 UI
- [ ] 대화 컨텍스트 전달 (현재 콘솔 시뮬레이션만 구현)

### 추가 개선
- [ ] A/B 테스트를 통한 임계값 최적화
- [ ] 주기적 DB 정리 및 최적화

---

## ✅ 완료된 항목

### Phase 1: 데이터 수집 및 DB 구축

| 항목 | 파일 |
|------|------|
| QnA 크롤링 | `src/crawlers/qna_crawler.py` |
| 이용약관 크롤링 | `src/crawlers/tos_crawler.py` |
| QnA Vector DB | `src/vectorstore/qna_store.py` |
| 약관 Vector DB + Chunker | `src/vectorstore/tos_store.py` |
| 약관 Graph DB | `src/graphstore/graph_store.py` |
| Triplet 추출 | `src/graphstore/triplet_extractor.py` |
| Hybrid Search | `src/retrieval/hybrid_search.py` |

### Phase 2: Hallucination Handling

| 항목 | 파일 |
|------|------|
| System Prompt 강화 | `src/verifier/prompts.py` |
| Citation 검증 | `src/verifier/verifier.py` |
| Verifier Agent | `src/verifier/verifier.py` |

### Phase 3: Fallback & 자동 학습

| 항목 | 파일 |
|------|------|
| 계층적 Fallback 시스템 | `src/router/router.py` |
| QnA/ToS Retriever | `src/retrieval/qna_retriever.py`, `tos_retriever.py` |
| Confidence 임계값 설정 | `configs/agent_config.yaml` |
| 상담원 답변 → QnA 자동 추가 | `src/feedback/feedback.py` |
| 중복 질문 감지 | `src/feedback/feedback.py` |