# Model Context Protocol (MCP) Server

This project includes a fully functional **MCP Server** (`src/mcp/server.py`) that allows external agents (specifically **Claude Desktop**) to interact with the RAG pipeline as a set of tools.

## 🔌 What is MCP?
The [Model Context Protocol](https://modelcontextprotocol.io/) is a standard for exposing data and functionality to AI models. It allows Claude to "call" functions in this codebase.

## 🛠️ Exposed Tools

The server exposes the following tools to the MCP client:

1.  **`ask_question(query: str)`**:
    *   The main entry point. Runs the full RAG pipeline (QnA -> ToS -> Verify).
    *   Returns the final answer and sources.
2.  **`search_faq(query: str)`**:
    *   Directly searches only the QnA database.
3.  **`search_terms(query: str)`**:
    *   Directly searches the Terms of Service database (Hybrid search).
4.  **`get_section(document: str, section: str)`**:
    *   Retrieves a specific section of a document (e.g., "Article 5" of "Service Terms").

## ⚙️ Configuration for Claude Desktop

To use this with the Claude Desktop App:

1.  Locate your config file:
    *   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    *   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2.  Add the server configuration:

```json
{
  "mcpServers": {
    "tiny-chatbot": {
      "command": "/path/to/your/python",
      "args": [
        "-m",
        "src.mcp.server"
      ],
      "cwd": "/path/to/project/root",
      "env": {
        "OPENAI_API_KEY": "your-key-here" 
      }
    }
  }
}
```

*Note: Ensure the python path points to the virtual environment (`.venv/bin/python`) where dependencies are installed.*
