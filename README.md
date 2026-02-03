# Tiny Chatbot Agents

**Local LLM-based Customer Service Chatbot for Terms of Service (ToS) and QnA.**

This project implements a robust Retrieval-Augmented Generation (RAG) pipeline designed to answer customer inquiries by first checking a QnA (FAQ) database and falling back to a detailed Terms of Service (ToS) document search if no matching FAQ is found. It features hybrid search, re-ranking, and hallucination verification to ensure high-quality answers.

## 🌟 Key Features

-   **Dual-Stage RAG Pipeline**: Prioritizes curated QnA matches before searching complex ToS documents.
-   **Hybrid Search**: Combines Vector Search (Semantic) with Rule-based matching and Knowledge Graph triplets for high precision.
-   **Advanced Ranking**: Uses **Bi-Encoder** (E5) for fast retrieval and **Cross-Encoder** (BGE-Reranker) for precise re-ranking.
-   **Hallucination Verification**: LLM-based verification step to ensure answers are grounded in the retrieved context.
-   **MCP Server Support**: Implements the [Model Context Protocol](https://modelcontextprotocol.io/) to integrate with Claude Desktop and other MCP clients.
-   **Local LLM Ready**: Designed to work with local inference servers (vLLM, Ollama) via OpenAI-compatible APIs.
-   **Robust Evaluation**: Integrated LLM-as-a-Judge framework with parallel execution and Korean-specific metrics.

## 🏗️ Architecture Overview

```mermaid
graph TD
    User[User Query] --> Pipeline[RAG Pipeline]
    
    subgraph "Stage 1: QnA Retrieval"
        Pipeline --> QnASearch["Vector Search (QnA DB)"]
        QnASearch -- "Score >= 0.80" --> QnAMatch[Found FAQ Match]
        QnAMatch --> LLM[LLM Response Generation]
    end
    
    subgraph "Stage 2: ToS Retrieval (Fallback)"
        QnASearch -- "Score < 0.80" --> ToSSearch["Hybrid Search (ToS DB)"]
        ToSSearch --> Vector[Vector Search]
        ToSSearch --> Keyword[Rule/Keyword Match]
        ToSSearch --> Reranker[Cross-Encoder Reranker]
        Reranker --> ToSContext[Selected ToS Sections]
        ToSContext --> LLM
    end
    
    subgraph "Stage 3: Verification"
        LLM --> Verifier[Hallucination Verifier]
        Verifier -- Verified --> Final[Final Answer]
        Verifier -- Issues Found --> Fallback[Uncertainty Response]
    end
```

## 🚀 Getting Started

### Prerequisites

-   Python 3.11 or higher
-   [uv](https://github.com/astral-sh/uv) (recommended) or pip
-   (Optional) A running Local LLM server (e.g., vLLM, Ollama) or OpenAI API Key.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/tiny-chatbot-agents.git
    cd tiny-chatbot-agents
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    # Create virtual environment and install packages
    uv sync
    ```
    *Or using pip:*
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

### Configuration

Configuration files are located in the `configs/` directory:

-   `agent_config.yaml`: Core agent settings (thresholds, LLM provider, search parameters).
-   `embedding_config.yaml`: Settings for embedding models (e.g., `multilingual-e5-large`) and rerankers.

## 💻 Usage

### 1. Running the Pipeline (CLI)
You can run the pipeline interactively or via scripts.
*(Check `main.py` or `scripts/run_pipeline.py` for entry points - custom implementation required based on specific needs)*

### 2. Web Interface (Streamlit)
To interact with the chatbot via a web interface:

```bash
streamlit run streamlit_app.py
```

This interface allows you to:
- Chat with the RAG pipeline.
- Search the QnA and ToS databases independently.
- Configure LLM providers and verification settings dynamically.

<p align="center">
  <img src="assets/3.png" width="32%">
  <img src="assets/4.png" width="32%">
  <img src="assets/5.png" width="32%">
</p>

### 3. Model Context Protocol (MCP) Server
To use this agent within Claude Desktop:

1.  **Run the MCP Server:**
    ```bash
    python scripts/run_mcp_server.py
    ```

2.  **Configure Claude Desktop:**
    Add the following to your `claude_desktop_config.json`:
    ```json
    {
      "mcpServers": {
        "rag-chatbot": {
          "command": "/absolute/path/to/venv/bin/python",
          "args": ["-m", "src.mcp.server"],
          "cwd": "/absolute/path/to/tiny-chatbot-agents",
          "env": {
            "OPENAI_API_KEY": "sk-..."
          }
        }
      }
    }
    ```

### 4. Data Ingestion
Populate your vector databases with crawled data:

```bash
# Crawl data
python scripts/crawl.py

# Ingest into Vector DB
python scripts/ingest_qna.py
python scripts/ingest_tos.py
```

## 📁 Project Structure

```
tiny-chatbot-agents/
├── configs/              # Configuration files (Agent, Embeddings)
├── data/                 # Data storage (VectorDB, Raw JSONs)
├── docs/                 # Documentation
├── scripts/              # Utility scripts (Crawling, Ingestion, MCP)
├── src/
│   ├── crawlers/         # Playwright-based web crawlers
│   ├── evaluation/       # LLM Judge evaluation framework
│   ├── llm/              # LLM Client wrappers
│   ├── mcp/              # MCP Server implementation
│   ├── pipeline/         # Core RAG Pipeline logic
│   ├── tos_search/       # Hybrid search & Reranking logic
│   ├── vectorstore/      # ChromaDB wrappers
│   └── verifier/         # Hallucination verification logic
└── tests/                # Unit tests
```

## 🛠️ Tech Stack

-   **Vector DB**: [ChromaDB](https://www.trychroma.com/)
-   **Embeddings**: `intfloat/multilingual-e5-large` (Bi-Encoder)
-   **Reranking**: `BAAI/bge-reranker-v2-m3` (Cross-Encoder)
-   **LLM Interface**: OpenAI API compatible (vLLM/Ollama support)
-   **MCP**: `fastmcp`
