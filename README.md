# Aircall Call Context Agent

AI agent for searching and analyzing Aircall call data using RAG (Retrieval-Augmented Generation), LangChain, and MCP (Model Context Protocol).

## Features

- ✅ **RAG Architecture** with ChromaDB vector store
- ✅ **LangChain** framework for agent orchestration
- ✅ **Reranking** for improved retrieval accuracy
- ✅ **Performance Optimization** (caching, streaming)
- ✅ **MCP Server** exposing call search tools
- ✅ **Fake call data** for demonstration

## Architecture

```
[Fake Call Data (JSON)] 
    ↓
[ChromaDB Vector Store] 
    ↓
[LangChain RAG Agent]
  - Retriever (top-k similarity)
  - Reranker (LLM-based relevance)
  - Generator (GPT-4)
    ↓
[MCP Server (exposes tools)]
    ↓
[MCP Client (Claude/GPT)]
```

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```bash
# Clone the repo
cd aircall-context-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Generate Data

```bash
python generate_calls.py
```

This creates fake call data in `data/calls.json` (20 sample calls).

### Ingest Data into Vector Store

```bash
python ingest.py
```

This embeds the calls and stores them in ChromaDB (`./chroma_db/`).

## Usage

### Basic RAG Agent

```bash
python agent.py
```

This runs sample queries against the call data.

### With Reranking

```bash
python reranker.py
```

This adds LLM-based reranking to improve retrieval accuracy.

### Optimized Agent

```bash
python optimized_agent.py
```

This includes caching and streaming for better performance.

### MCP Server

Start the MCP server:
```bash
python mcp_server.py
```

In another terminal, test with the MCP client:
```bash
python mcp_client.py
```

## Example Queries

- "What billing issues have customers reported?"
- "Which agent handled the most calls?"
- "What call quality problems were mentioned?"
- "Show me calls from the last week"

## Tech Stack

- **LangChain** 0.1.x - AI agent framework
- **ChromaDB** 0.4.x - Vector database
- **OpenAI API** - GPT-4 for generation, embeddings
- **MCP SDK** - Model Context Protocol
- **Python** 3.11+

## Project Structure

```
aircall-context-agent/
├── README.md
├── requirements.txt
├── .env.example
├── plan.md                  # Build plan document
├── generate_calls.py        # Generate fake call data
├── ingest.py                # Embed and store in ChromaDB
├── agent.py                 # Basic RAG agent
├── reranker.py              # RAG with reranking
├── optimized_agent.py       # Performance-optimized version
├── mcp_server.py            # MCP server implementation
├── mcp_client.py            # MCP client test
├── data/
│   └── calls.json           # Generated call data
└── chroma_db/               # ChromaDB vector store (created by ingest.py)
```

## Why This Project?

This demonstrates hands-on experience with:

- **LangChain**: Framework for building AI agents
- **RAG**: Combining retrieval with generation for better accuracy
- **Vector Databases**: Semantic search over unstructured data
- **Reranking**: Improving retrieval relevance beyond vector similarity
- **MCP**: Modern protocol for AI-tool integration
- **Performance Optimization**: Caching, streaming, batch processing

## Next Steps

- Add more sophisticated reranking (Cohere Rerank API, cross-encoders)
- Implement caching layer (Redis)
- Add filtering by metadata (date range, agent, tags)
- Build web UI for querying
- Deploy to AWS Lambda + API Gateway

## License

MIT
