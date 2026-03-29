# Documentation Overview

Welcome to the **Tiny Chatbot Agents** documentation. The current runtime uses a LangGraph-based RAG pipeline for customer-service inquiries over FAQ and Terms of Service data.

## 📚 Contents

1.  [**RAG Architecture**](rag_architecture.md): LangGraph node flow, routing thresholds, and verification behavior.
2.  [**Vector Search System**](vector_store.md): Deep dive into ChromaDB, E5 embeddings, and schema design.
3.  [**Search & Ranking**](search_ranking.md): Explanation of Hybrid Search, Rule Matchers, and Cross-Encoder Reranking.
4.  [**MCP Server**](mcp_server.md): Current MCP tools and `main.py mcp` integration.
5.  [**Evaluation**](evaluation.md): Unified CLI evaluation flow via `python main.py evaluate`.
6.  [**Crawlers**](crawlers.md): Crawl and ingest workflow via the unified CLI.
