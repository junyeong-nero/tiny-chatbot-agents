"""MCP Server for RAG Pipeline.

Exposes the following tools to MCP clients:
- ask_question: Full RAG pipeline (QnA -> ToS -> Verify)
- search_faq: Direct QnA database search
- search_terms: Direct ToS database search (hybrid)
- get_section: Retrieve specific section from a document
- list_documents: List available documents

Usage:
    # Run directly
    python -m src.mcp.server

    # Or use the script
    python scripts/run_mcp_server.py

For Claude Desktop configuration, see docs/mcp_server.md.
"""

import logging
from typing import Any

from fastmcp import FastMCP

from src.llm import create_llm_client
from src.pipeline import RAGPipeline
from src.vectorstore import QnAVectorStore, ToSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="tiny-chatbot",
    instructions="""
    This is a Korean financial services customer support chatbot.
    It can answer questions about FAQs and Terms of Service.
    Use ask_question for general queries, or search_faq/search_terms for specific searches.
    """,
)

# Lazy initialization of pipeline components
_pipeline: RAGPipeline | None = None
_qna_store: QnAVectorStore | None = None
_tos_store: ToSVectorStore | None = None


def _get_pipeline() -> RAGPipeline:
    """Get or initialize the RAG pipeline."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing RAG Pipeline...")
        _pipeline = RAGPipeline(
            enable_verification=True,
            enable_hybrid_tos_search=True,
        )
        logger.info("RAG Pipeline initialized successfully")
    return _pipeline


def _get_qna_store() -> QnAVectorStore:
    """Get or initialize the QnA store."""
    global _qna_store
    if _qna_store is None:
        _qna_store = QnAVectorStore(persist_directory="data/vectordb/qna")
    return _qna_store


def _get_tos_store() -> ToSVectorStore:
    """Get or initialize the ToS store."""
    global _tos_store
    if _tos_store is None:
        _tos_store = ToSVectorStore(
            persist_directory="data/vectordb/tos",
            enable_hybrid_search=True,
        )
    return _tos_store


@mcp.tool
def ask_question(query: str) -> dict[str, Any]:
    """Ask a question to the customer service chatbot.

    Runs the full RAG pipeline:
    1. Search QnA database for similar questions
    2. If no match, search Terms of Service
    3. Verify the answer for hallucinations

    Args:
        query: The customer's question in Korean

    Returns:
        A dictionary containing:
        - answer: The generated answer
        - source: Where the answer came from (qna, tos, no_context)
        - confidence: Confidence score (0-1)
        - citations: List of source citations
        - verified: Whether the answer passed verification
    """
    pipeline = _get_pipeline()
    response = pipeline.query(query)

    return {
        "answer": response.answer,
        "source": response.source.value,
        "confidence": round(response.confidence, 3),
        "response_mode": response.response_mode,
        "citations": response.citations,
        "verified": response.verified,
        "verification_score": round(response.verification_score, 3),
        "verification_issues": response.verification_issues,
    }


@mcp.tool
def search_faq(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Search the FAQ database directly.

    Performs a semantic search on the QnA database without LLM generation.
    Useful for finding similar questions or checking if an answer exists.

    Args:
        query: Search query (Korean)
        n_results: Maximum number of results to return (default: 5)

    Returns:
        List of matching FAQ entries with:
        - question: The FAQ question
        - answer: The FAQ answer
        - category: Question category
        - score: Similarity score (0-1)
    """
    pipeline = _get_pipeline()
    results = pipeline.search_qna(query, n_results=min(n_results, 10))
    return results


@mcp.tool
def search_terms(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Search the Terms of Service database directly.

    Performs a hybrid search (vector + rule-based) on the ToS database.
    Useful for finding specific clauses or regulations.

    Args:
        query: Search query (Korean), can include section references like "제1조"
        n_results: Maximum number of results to return (default: 5)

    Returns:
        List of matching ToS sections with:
        - document_title: Name of the document
        - section_title: Section heading (e.g., "제1조")
        - section_content: Full text of the section
        - score: Relevance score (0-1)
    """
    pipeline = _get_pipeline()
    results = pipeline.search_tos(query, n_results=min(n_results, 10))
    return results


@mcp.tool
def get_section(document: str, section: str) -> dict[str, Any] | None:
    """Retrieve a specific section from a document.

    Args:
        document: Document name (e.g., "서비스이용약관", "개인정보처리방침")
        section: Section reference (e.g., "제1조", "제5조 제2항")

    Returns:
        Section details if found, None otherwise:
        - document_title: Document name
        - section_title: Full section title
        - section_content: Section text
    """
    tos_store = _get_tos_store()

    # Search for the specific section
    query = f"{document} {section}"
    results = tos_store.search(query, n_results=5)

    if not results:
        return None

    # Find exact or best match
    for result in results:
        if section in (result.section_title or ""):
            return {
                "document_title": result.document_title,
                "section_title": result.section_title,
                "section_content": result.section_content,
            }

    # Return best match if no exact match
    best = results[0]
    return {
        "document_title": best.document_title,
        "section_title": best.section_title,
        "section_content": best.section_content,
        "note": "Exact section not found, returning best match",
    }


@mcp.tool
def list_documents() -> dict[str, Any]:
    """List all available documents in the knowledge base.

    Returns:
        Dictionary with:
        - qna_count: Number of FAQ entries
        - tos_documents: List of Terms of Service documents
        - categories: List of FAQ categories
    """
    qna_store = _get_qna_store()
    tos_store = _get_tos_store()

    # Get QnA categories
    qna_count = qna_store.count()

    # Get ToS document titles
    tos_results = tos_store.search("약관", n_results=50)
    tos_documents = list({r.document_title for r in tos_results if r.document_title})

    return {
        "qna_count": qna_count,
        "tos_document_count": tos_store.count(),
        "tos_documents": tos_documents,
    }


@mcp.tool
def health_check() -> dict[str, Any]:
    """Check the health of the RAG system.

    Verifies that all components are properly initialized and accessible.

    Returns:
        Health status for each component:
        - llm: LLM connection status
        - qna_store: QnA database status
        - tos_store: ToS database status
        - overall: Overall system health
    """
    status = {
        "llm": False,
        "qna_store": False,
        "tos_store": False,
        "overall": False,
    }

    try:
        pipeline = _get_pipeline()

        # Check LLM
        if hasattr(pipeline.llm, "health_check"):
            status["llm"] = pipeline.llm.health_check()
        else:
            # For OpenAI client, assume healthy if initialized
            status["llm"] = True

        # Check stores
        status["qna_store"] = pipeline.qna_store.count() > 0
        status["tos_store"] = pipeline.tos_store.count() > 0

        status["overall"] = all([status["llm"], status["qna_store"], status["tos_store"]])

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        status["error"] = str(e)

    return status


if __name__ == "__main__":
    logger.info("Starting MCP Server for RAG Pipeline...")
    logger.info("Available tools: ask_question, search_faq, search_terms, get_section, list_documents, health_check")
    mcp.run()
