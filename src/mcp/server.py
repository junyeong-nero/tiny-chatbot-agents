"""FastMCP Server for RAG Pipeline.

Exposes QnA and ToS search tools via MCP protocol.

Usage:
    # Run directly
    python -m src.mcp.server

    # Or use with Claude Desktop / other MCP clients
    Add to claude_desktop_config.json:
    {
        "mcpServers": {
            "rag-chatbot": {
                "command": "python",
                "args": ["-m", "src.mcp.server"]
            }
        }
    }
"""

import logging
from typing import Annotated

from fastmcp import FastMCP

from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Global pipeline instance (lazy initialized)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def create_mcp_server(name: str = "rag-chatbot") -> FastMCP:
    """Create and configure the MCP server.

    Args:
        name: Server name

    Returns:
        Configured FastMCP server
    """
    mcp = FastMCP(name)

    @mcp.tool()
    def ask_question(
        question: Annotated[str, "사용자의 질문. 예: '계좌 해지 방법이 뭐야?'"],
    ) -> str:
        """사용자 질문에 대해 QnA DB와 ToS DB를 검색하여 답변합니다.

        먼저 FAQ(QnA) DB에서 유사한 질문을 찾고,
        없으면 약관(ToS) DB에서 관련 조항을 찾아 답변을 생성합니다.
        """
        pipeline = get_pipeline()
        response = pipeline.query(question)

        result = f"[출처: {response.source.value.upper()}]\n\n"
        result += response.answer

        if response.citations:
            result += f"\n\n참조: {', '.join(response.citations)}"

        return result

    @mcp.tool()
    def search_faq(
        query: Annotated[str, "검색할 질문. 예: '비밀번호 변경'"],
        n_results: Annotated[int, "반환할 결과 수"] = 5,
    ) -> str:
        """FAQ(QnA) DB에서 유사한 질문을 검색합니다.

        질문과 답변을 함께 반환합니다.
        """
        pipeline = get_pipeline()
        results = pipeline.search_qna(query, n_results=n_results)

        if not results:
            return "관련된 FAQ를 찾을 수 없습니다."

        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"[{i}] (유사도: {r['score']:.2f})\n"
                f"Q: {r['question']}\n"
                f"A: {r['answer']}\n"
                f"카테고리: {r['category']} > {r['sub_category']}"
            )

        return "\n\n---\n\n".join(output)

    @mcp.tool()
    def search_terms(
        query: Annotated[str, "검색할 내용. 예: '해지 환불' 또는 '제3조'"],
        n_results: Annotated[int, "반환할 결과 수"] = 5,
    ) -> str:
        """약관(ToS) DB에서 관련 조항을 검색합니다.

        특정 조항을 검색하거나 (예: '제1조 1항'),
        키워드로 관련 조항을 찾을 수 있습니다 (예: '해지 환불').
        """
        pipeline = get_pipeline()
        results = pipeline.search_tos(query, n_results=n_results)

        if not results:
            return "관련된 약관 조항을 찾을 수 없습니다."

        output = []
        for i, r in enumerate(results, 1):
            content = r["section_content"]
            if len(content) > 500:
                content = content[:500] + "..."

            output.append(
                f"[{i}] (유사도: {r['score']:.2f})\n"
                f"약관: {r['document_title']}\n"
                f"조항: {r['section_title']}\n"
                f"내용: {content}"
            )

        return "\n\n---\n\n".join(output)

    @mcp.tool()
    def get_section(
        section: Annotated[str, "조항 패턴. 예: '제1조', '제3조 2항'"],
        document: Annotated[str | None, "약관 이름 필터 (선택)"] = None,
    ) -> str:
        """특정 약관 조항의 전체 내용을 조회합니다.

        예: '제1조 1항에 대해 알려줘' → section='제1조'
        """
        pipeline = get_pipeline()
        results = pipeline.get_tos_section(
            document_title=document, section_pattern=section
        )

        if not results:
            return f"'{section}' 조항을 찾을 수 없습니다."

        output = []
        for r in results:
            output.append(
                f"[{r['document_title']}]\n"
                f"{r['section_title']}\n\n"
                f"{r['section_content']}\n\n"
                f"시행일: {r['effective_date']}"
            )

        return "\n\n===\n\n".join(output)

    @mcp.tool()
    def list_documents() -> str:
        """사용 가능한 약관 목록을 조회합니다."""
        pipeline = get_pipeline()

        # Get unique document titles from a broad search
        results = pipeline.search_tos("약관", n_results=100)

        documents = set()
        for r in results:
            documents.add(r["document_title"])

        if not documents:
            return "등록된 약관이 없습니다."

        output = ["=== 등록된 약관 목록 ===\n"]
        for doc in sorted(documents):
            output.append(f"• {doc}")

        return "\n".join(output)

    return mcp


# Create server instance
mcp = create_mcp_server()

if __name__ == "__main__":
    mcp.run()
