# LangGraph Migration Validation

## Commands Run

### Unit + graph tests
```bash
uv run pytest tests/test_pipeline.py tests/test_graph_nodes.py tests/test_graph_routers.py
```

Result:
```text
36 passed in 4.71s
```

### Integration tests
```bash
uv run pytest tests/test_pipeline_integration.py
```

Result:
```text
33 passed, 1 skipped, 4 warnings in 674.75s (0:11:14)
```

### Coverage over migration-relevant files
```bash
uv run --with coverage python -m coverage run -m pytest tests/test_pipeline.py tests/test_graph_nodes.py tests/test_graph_routers.py tests/test_pipeline_integration.py
uv run --with coverage python -m coverage report -m src/pipeline/rag_pipeline.py src/graph/edges/routers.py src/graph/graph.py src/graph/nodes/search.py src/graph/nodes/generate.py src/graph/nodes/verify.py src/graph/nodes/format.py src/graph/state.py src/graph/utils.py src/pipeline/models.py
```

Result:
```text
69 passed, 1 skipped, 4 warnings in 369.29s (0:06:09)

Name                           Stmts   Miss  Cover
--------------------------------------------------
src/graph/edges/routers.py        15      0   100%
src/graph/graph.py                24      0   100%
src/graph/nodes/format.py          5      0   100%
src/graph/nodes/generate.py       93      9    90%
src/graph/nodes/search.py         30      7    77%
src/graph/nodes/verify.py         24      3    88%
src/graph/state.py                24      0   100%
src/graph/utils.py                18      0   100%
src/pipeline/models.py            22      0   100%
src/pipeline/rag_pipeline.py      66      9    86%
--------------------------------------------------
TOTAL                            321     28    91%
```

## Manual QA

Command:
```bash
uv run python - <<'PY'
from unittest.mock import Mock
from src.pipeline.rag_pipeline import RAGPipeline

class MockQnAResult:
    def __init__(self, question, answer, score, category='FAQ', sub_category='일반'):
        self.question = question
        self.answer = answer
        self.score = score
        self.category = category
        self.sub_category = sub_category
        self.source = 'FAQ'
        self.source_url = 'http://example.com'
        self.id = 'qna-1'

class MockToSResult:
    def __init__(self, document_title, section_title, section_content, score, category='약관'):
        self.document_title = document_title
        self.section_title = section_title
        self.section_content = section_content
        self.score = score
        self.category = category
        self.parent_content = ''
        self.effective_date = '2024-01-01'
        self.source_url = 'http://example.com'
        self.id = 'tos-1'

class MockLLMResponse:
    def __init__(self, content):
        self.content = content
        self.model = 'mock-llm'
        self.usage = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}

llm = Mock()
llm.generate_with_context.side_effect = [
    MockLLMResponse('FAQ 고신뢰 답변'),
    MockLLMResponse('FAQ 중간대역 제한 답변'),
    MockLLMResponse('약관 답변 [참조: 제1조]'),
]

qna_store = Mock()
tos_store = Mock()
qna_store.count.return_value = 2
tos_store.count.return_value = 2
qna_store.search.side_effect = [
    [MockQnAResult('비밀번호 변경', '설정에서 변경 가능합니다.', 0.91)],
    [MockQnAResult('유사 질문', '유사 답변', 0.75)],
    [],
    [MockQnAResult('별도 검색', '검색 답변', 0.5)],
]
tos_store.search.side_effect = [
    [MockToSResult('이용약관', '제1조 (목적)', '본 약관은...', 0.82)],
    [MockToSResult('이용약관', '제1조 (목적)', '본 약관은...', 0.82)],
]

pipeline = RAGPipeline(llm=llm, qna_store=qna_store, tos_store=tos_store, enable_verification=False)

responses = [
    pipeline.query('비밀번호 어떻게 바꿔요?'),
    pipeline.query('애매한 FAQ 질문'),
    pipeline.query('제1조 알려줘'),
]
for response in responses:
    print({'source': response.source.value, 'mode': response.response_mode, 'confidence': response.confidence, 'answer': response.answer})

print({'top_k_qna_len': len(pipeline.search_qna('질문', top_k=1))})
tos_store.search.return_value = [MockToSResult('이용약관', '제2조', '내용', 0.4)]
print({'top_k_tos_len': len(pipeline.search_tos('질문', top_k=1))})
PY
```

Observed output:
```text
{'source': 'qna', 'mode': 'answer', 'confidence': 0.91, 'answer': 'FAQ 고신뢰 답변'}
{'source': 'qna', 'mode': 'limited_answer', 'confidence': 0.75, 'answer': 'FAQ 중간대역 제한 답변'}
{'source': 'tos', 'mode': 'answer', 'confidence': 0.82, 'answer': '약관 답변 [참조: 제1조]'}
{'top_k_qna_len': 1}
{'top_k_tos_len': 1}
```

## Summary

- Graph routing matches the planned QnA high/mid/low and ToS high/mid/low branching.
- `RAGPipeline` remains the facade and preserves caller compatibility, including `top_k` aliases used elsewhere in the repo.
- Integration tests pass after the migration.
- Coverage across the migration-relevant files is `91%`, above the `80%` target in `plan.md`.
