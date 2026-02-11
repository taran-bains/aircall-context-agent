# Aircall Call Context Agent

AI agent for searching and analyzing Aircall call data using RAG (Retrieval-Augmented Generation), LangChain, and MCP (Model Context Protocol).

## Features

- ✅ **RAG Architecture** with ChromaDB vector store
- ✅ **LangChain** framework for agent orchestration
- ✅ **Reranking** for improved retrieval accuracy
- ✅ **Performance Optimization** (streaming, caching)
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
  - Generator (Claude 4.5 Haiku)
  - Embeddings (Local: all-MiniLM-L6-v2)
    ↓
[MCP Server (exposes tools)]
    ↓
[MCP Client (Claude/GPT)]
```

## Setup

### Prerequisites

- Python 3.11+
- Anthropic API key (for Claude 4.5 Haiku)

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
# Add your ANTHROPIC_API_KEY to .env
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

### Optimized Agent (with Streaming)

```bash
python optimized_agent.py
```

This demonstrates **true token-by-token streaming**:
- Bypasses RetrievalQA's buffering by manually retrieving docs and streaming from LLM
- Tokens appear in real-time as Claude generates them
- Much lower perceived latency compared to waiting for full response

**Streaming modes:**
- **Normal**: `python optimized_agent.py` - Full speed streaming
- **Slow motion**: `python optimized_agent.py --slow` - Adds 10ms delay between tokens to visually demonstrate streaming

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

### Best Suited for RAG (Semantic Search):
- "What billing issues have customers reported?"
- "What call quality problems were mentioned?"
- "Show me calls about integration issues"
- "What technical problems have been escalated?"

### Limited Accuracy (Requires Full Dataset):
- "Which agent handled the most calls?" ⚠️
- "How many unresolved issues are there?" ⚠️
- "Show me all calls from the last week" ⚠️

**Note:** RAG retrieves the top 10-15 most relevant documents, which is excellent for semantic search but may not provide accurate counts/aggregations across the entire dataset.

## Tech Stack

- **LangChain** 0.1.x - AI agent framework
- **ChromaDB** 0.4.x - Vector database
- **Anthropic API** - Claude 4.5 Haiku for generation and reranking
- **HuggingFace** - Local embeddings (`all-MiniLM-L6-v2`)
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

## Architecture Considerations

### When RAG Works Well ✅
- **Semantic search queries**: Finding documents by meaning, not exact keywords
- **Exploratory questions**: "What issues were reported about X?"
- **Context-based answers**: Synthesizing information from relevant documents

### When RAG Has Limitations ⚠️
- **Counting/aggregation**: "How many calls did each agent handle?"
- **Exhaustive listings**: "Show me ALL unresolved issues"
- **Statistical analysis**: Requires seeing the full dataset

### Solution Approaches
For production systems, you'd typically use:
1. **RAG for semantic search** (as implemented here)
2. **Direct database queries** for counting/aggregation
3. **Hybrid approach** where the LLM decides which tool to use based on the query type

The MCP server's `get_call_stats` tool demonstrates this hybrid approach.

## Why This Project?

This demonstrates hands-on experience with:

- **LangChain**: Framework for building AI agents
- **RAG**: Combining retrieval with generation for better accuracy
- **Local Embeddings**: Privacy-first, cost-efficient semantic search using local models
- **Claude 4.5 Haiku**: High-performance, fast, and cost-efficient generation
- **Reranking**: Improving retrieval relevance beyond vector similarity
- **MCP**: Modern protocol for AI-tool integration
- **Performance Optimization**: Token streaming for lower latency, LLM response caching
- **Understanding Trade-offs**: Knowing when RAG works and when it doesn't

## Next Steps

- Add more sophisticated reranking (Cohere Rerank API, cross-encoders)
- Implement caching layer (Redis)
- Add filtering by metadata (date range, agent, tags)
- Build web UI for querying
- Deploy to AWS Lambda + API Gateway

## License

MIT
