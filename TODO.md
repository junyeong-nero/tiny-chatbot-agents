기능 제안 (코드 기준 갭/연결 포인트 포함)
- Adaptive Thresholding: 현재 고정 임계값(DEFAULT_QNA_THRESHOLD, DEFAULT_TOS_THRESHOLD). 중간 영역은 “근거 제한 답변/재질문/상담원 연결”로 분기하는 게 안전함: src/pipeline/rag_pipeline.py
- HyDE/Query Expansion: ToS는 장문의 법률 문체라 짧은 질의 매칭이 약함. src/pipeline/rag_pipeline.py에서 ToS 검색 직전 질의 확장 후 검색(또는 병렬 검색) 추가 권장.
- Citation‑to‑Context 매핑 강화: 현재 인용 패턴 매칭은 섹션 제목 기반 부분 일치. 조항 번호/제목 표준화 테이블을 만들어 안정성 향상: src/verifier/verifier.py, src/tos_store.py
- Chunking 고도화: ToSChunker가 섹션 기반이긴 하나 최대 길이 절단과 parent_context는 고정. 문장 경계/오버랩 기반 semantic chunking 추가하면 정확도 개선: src/vectorstore/tos_store.py
- Query Normalization (Korean): 조사/종결어미 정규화, 숫자/조항 패턴 정규화 전처리를 추가해 검색 품질을 올릴 수 있음: src/pipeline/rag_pipeline.py, src/tos_search/rule_matcher.py
- Telemetry/Trace: 응답마다 retrieval scores, 선택된 문서, verification 결과를 구조화 로그로 남기면 운영 진단이 쉬움: src/pipeline/rag_pipeline.py
- Human‑in‑the‑loop QnA 확장: README에 “상담원 답변 자동 추가”가 있지만 코드 흐름은 보이지 않음. QnA 백필 파이프라인 추가 여지가 큼: src/vectorstore/qna_store.py, scripts/ingest_qna.py
