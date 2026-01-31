"""MCP Server module for RAG Pipeline.

Exposes the RAG pipeline as MCP tools for external agents like Claude Desktop.
"""

from .server import mcp

__all__ = ["mcp"]
