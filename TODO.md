# TODO

## Evaluation Pipeline 개선 (Priority Order)

### 🔴 Critical

- [x] **1. 한국어 토크나이저 도입**: BLEU 계산 시 space split 대신 형태소 분석기 사용. 현재 "한국투자증권에서"가 단일 토큰으로 처리되어 n-gram 매칭이 부정확함
  - 파일: `src/evaluation/evaluator.py:156-217`
  - 방안: kiwipiepy (경량, 순수 Python) 또는 konlpy 사용
  - **완료**: kiwipiepy 도입, singleton 패턴으로 메모리 효율화

- [x] **2. 배치 평가 병렬 처리**: 50개 테스트 케이스 + LLM Judge 사용 시 순차 처리로 시간 과다 소요
  - 파일: `src/evaluation/runner.py:275-315`
  - 방안: asyncio + ThreadPoolExecutor 또는 concurrent.futures 활용
  - **완료**: ThreadPoolExecutor 기반 병렬 실행, `--parallel` / `--max-workers` CLI 옵션 추가

- [x] **3. LLM Judge JSON 파싱 안정화**: 파싱 실패 시 0.0 점수 부여가 결과 왜곡 유발
  - 파일: `src/evaluation/llm_judge.py:244-291`
  - 방안: Retry 로직, Partial parsing, Structured output mode
  - **완료**: Exponential backoff retry, regex fallback 파싱, 에러 시 neutral score (3.0) 사용

### 🟡 High Priority

- [x] **4. Context Overlap 메트릭 활성화**: `compute_context_overlap` 정의되어 있으나 미사용
  - 파일: `src/evaluation/evaluator.py:249-285`
  - 방안: EvaluationMetrics에 context_recall/precision 추가
  - **완료**: context_recall, context_precision 필드 추가, evaluate()에서 expected_sources 지원

- [x] **5. Embedding Model 싱글톤화**: 각 Evaluator가 별도 모델 로드로 메모리 낭비
  - 파일: `src/evaluation/evaluator.py:121-132`
  - **완료**: `_get_embedding_model()` 싱글톤 패턴 도입

- [x] **6. Judge Model Diversity 기본값 강화**: strict_diversity=False가 기본, 같은 모델 평가 허용
  - 파일: `src/evaluation/llm_judge.py:432-440`
  - **완료**: `strict_diversity=True`로 기본값 변경

### 🟢 Medium Priority

- [x] **7. 메트릭 스케일 표준화**: similarity/bleu는 0-1, llm_judge는 1-5로 혼재
  - **완료**: 정규화된 LLM Judge 메트릭 추가 (mean_llm_*_normalized, 0-1 스케일)
- [x] **8. 테스트 커버리지 확대**: LLMJudge, FrontierClient 테스트 부재
  - **완료**: 39개 테스트 (LLMJudgeComprehensive, JudgeModelSelector, ContextOverlapMetrics 등)
- [x] **9. Dataset Schema Validation**: Pydantic 기반 검증 추가
  - **완료**: `src/evaluation/schemas.py` 추가, EvaluationTestCase/EvaluationDataset 모델
- [x] **10. Faithfulness 명칭 명확화**: verifier vs judge 구분 개선
  - **완료**: verifier_faithfulness vs judge_context_faithfulness 명명 규칙 적용

---

## RAG Pipeline 기능 제안

- Adaptive Thresholding: 현재 고정 임계값(DEFAULT_QNA_THRESHOLD, DEFAULT_TOS_THRESHOLD). 중간 영역은 "근거 제한 답변/재질문/상담원 연결"로 분기하는 게 안전함: src/pipeline/rag_pipeline.py
- HyDE/Query Expansion: ToS는 장문의 법률 문체라 짧은 질의 매칭이 약함. src/pipeline/rag_pipeline.py에서 ToS 검색 직전 질의 확장 후 검색(또는 병렬 검색) 추가 권장.
- Citation-to-Context 매핑 강화: 현재 인용 패턴 매칭은 섹션 제목 기반 부분 일치. 조항 번호/제목 표준화 테이블을 만들어 안정성 향상: src/verifier/verifier.py, src/tos_store.py
- Chunking 고도화: ToSChunker가 섹션 기반이긴 하나 최대 길이 절단과 parent_context는 고정. 문장 경계/오버랩 기반 semantic chunking 추가하면 정확도 개선: src/vectorstore/tos_store.py
- Query Normalization (Korean): 조사/종결어미 정규화, 숫자/조항 패턴 정규화 전처리를 추가해 검색 품질을 올릴 수 있음: src/pipeline/rag_pipeline.py, src/tos_search/rule_matcher.py
- Telemetry/Trace: 응답마다 retrieval scores, 선택된 문서, verification 결과를 구조화 로그로 남기면 운영 진단이 쉬움: src/pipeline/rag_pipeline.py
- [DONE] Human-in-the-loop QnA 확장: src/vectorstore/backfill.py + scripts/backfill_agent_answers.py 구현 완료
