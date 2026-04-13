# Tiny Chatbot Agents — 문서 목차

이 프로젝트는 LangGraph로 오케스트레이션되는 계층적 RAG(검색-증강 생성) 파이프라인 기반의 한국어 고객 서비스 챗봇입니다.

---

## 문서 목록

| 문서 | 설명 |
|------|------|
| [RAG 아키텍처](rag_architecture.md) | LangGraph 노드 흐름, 라우팅 임계값, 검증 동작 전체 개요 |
| [벡터 검색 시스템](vector_store.md) | ChromaDB 구성, E5 임베딩 모델, QnA/ToS 스키마 |
| [검색 및 랭킹](search_ranking.md) | 하이브리드 검색, 룰 매처, Cross-Encoder 리랭킹 원리 |
| [MCP 서버](mcp_server.md) | Claude Desktop 연동, 노출 툴 목록, 설정 방법 |
| [평가 프레임워크](evaluation.md) | LLM-as-a-Judge 평가 흐름, 메트릭, CLI 사용법 |
| [데이터 크롤러](crawlers.md) | 크롤링 및 벡터 DB 적재 워크플로 |

---

## 빠른 시작

처음 보는 분은 아래 순서로 읽는 것을 권장합니다.

1. **[RAG 아키텍처](rag_architecture.md)** — 전체 파이프라인 흐름과 라우팅 로직 이해
2. **[벡터 검색 시스템](vector_store.md)** — 데이터가 어떻게 저장·검색되는지 파악
3. **[검색 및 랭킹](search_ranking.md)** — ToS 하이브리드 검색의 점수 계산 방식
4. **[데이터 크롤러](crawlers.md)** — 직접 데이터를 수집하고 적재하는 방법

Claude Desktop 연동이 목적이라면 **[MCP 서버](mcp_server.md)** 를, 모델 성능 측정이 목적이라면 **[평가 프레임워크](evaluation.md)** 를 바로 참고하세요.

---

## 컴포넌트 의존 관계

```
main.py (CLI)
    │
    ├── RAGPipeline          ← src/pipeline/rag_pipeline.py
    │       │
    │       └── LangGraph    ← src/graph/graph.py
    │             ├── Nodes  ← src/graph/nodes/
    │             └── Edges  ← src/graph/edges/
    │
    ├── VectorStores         ← src/vectorstore/
    │       ├── QnAVectorStore
    │       └── ToSVectorStore
    │             └── HybridSearch  ← src/tos_search/
    │
    ├── AnswerVerifier       ← src/verifier/
    ├── LLMClient            ← src/llm/
    ├── MCPServer            ← src/mcp/
    └── Evaluation           ← src/evaluation/
```
