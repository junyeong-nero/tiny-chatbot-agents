#!/usr/bin/env python3
"""Run MCP Server for RAG Pipeline.

Usage:
    python scripts/run_mcp_server.py

For Claude Desktop, add to claude_desktop_config.json:
{
    "mcpServers": {
        "rag-chatbot": {
            "command": "python",
            "args": ["-m", "src.mcp.server"],
            "cwd": "/path/to/tiny-chatbot-agents",
            "env": {
                "OPENAI_API_KEY": "your-api-key"
            }
        }
    }
}
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mcp.server import mcp

if __name__ == "__main__":
    print("Starting MCP Server for RAG Pipeline...")
    print("Available tools: ask_question, search_faq, search_terms, get_section, list_documents")
    mcp.run()
