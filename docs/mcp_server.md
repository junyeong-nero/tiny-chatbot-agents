# MCP 서버

`src/mcp/server.py`는 [Model Context Protocol(MCP)](https://modelcontextprotocol.io/) 서버를 구현합니다. Claude Desktop 등 MCP 클라이언트가 RAG 파이프라인을 툴 형태로 호출할 수 있게 합니다.

---

## 노출 툴 목록

서버는 6개의 툴을 MCP 클라이언트에 노출합니다.

### `ask_question(query: str)`

전체 RAG 파이프라인을 실행합니다. QnA 검색 → ToS 폴백 → 환각 검증 순으로 처리됩니다.

```python
# 반환 예시
{
    "answer": "계좌 해지는 앱 > 내 계좌 > 해지 신청에서 가능합니다.",
    "source": "qna",                # "qna" | "tos" | "no_context"
    "confidence": 0.923,
    "response_mode": "answer",      # "answer" | "limited_answer" | "clarification" | "no_context"
    "citations": ["제15조"],
    "verified": True,
    "verification_score": 0.87,
    "verification_issues": []
}
```

### `search_faq(query: str, n_results: int = 5)`

LLM 생성 없이 QnA DB를 직접 검색합니다. 유사 질문이 있는지 확인할 때 유용합니다.

```python
# 반환: list of
{
    "question": "비밀번호를 잊어버렸어요",
    "answer": "...",
    "category": "보안",
    "score": 0.91
}
```

### `search_terms(query: str, n_results: int = 5)`

ToS DB를 하이브리드 검색합니다. `제1조`처럼 조항 번호를 포함한 쿼리도 지원합니다.

```python
# 반환: list of
{
    "document_title": "서비스이용약관",
    "section_title": "제1조 (목적)",
    "section_content": "...",
    "score": 0.84
}
```

### `get_section(document: str, section: str)`

특정 문서의 특정 조항을 조회합니다. 정확히 일치하는 조항이 없으면 가장 유사한 결과를 반환합니다.

```python
get_section(document="서비스이용약관", section="제5조")
# 반환: { "document_title": ..., "section_title": ..., "section_content": ... }
# 정확한 조항 미발견 시: "note": "Exact section not found, returning best match" 포함
```

### `list_documents()`

지식 베이스에 저장된 문서 현황을 반환합니다.

```python
{
    "qna_count": 342,
    "tos_document_count": 1580,
    "tos_documents": ["서비스이용약관", "개인정보처리방침", ...]
}
```

### `health_check()`

파이프라인 전체 컴포넌트의 상태를 점검합니다.

```python
{
    "llm": True,
    "qna_store": True,
    "tos_store": True,
    "overall": True
}
```

---

## 서버 실행

```bash
# CLI 방식 (권장)
python main.py mcp

# 직접 모듈 실행
python -m src.mcp.server
```

서버는 `fastmcp`를 사용하여 stdio 기반으로 실행됩니다. 파이프라인, QnA 스토어, ToS 스토어는 첫 번째 툴 호출 시 지연 초기화됩니다.

---

## Claude Desktop 연동

### 1. 설정 파일 위치

| OS | 경로 |
|----|------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

### 2. 서버 설정 추가

```json
{
  "mcpServers": {
    "tiny-chatbot": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/tiny-chatbot-agents",
      "env": {
        "LLM_PROVIDER": "vllm",
        "VLLM_API_BASE": "http://localhost:8000/v1",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

> `command`는 반드시 프로젝트 가상환경의 Python 경로여야 합니다 (`.venv/bin/python`).

### 3. 동작 확인

Claude Desktop에서 다음처럼 사용할 수 있습니다.

```
계좌 해지 방법이 뭐야?
→ ask_question("계좌 해지 방법이 뭐야?") 자동 호출
```
