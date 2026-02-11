# Aircall Call Context Agent - Build Plan
## Minimal RAG + MCP Demo to Fill Resume Gaps

**Goal:** Demonstrate LangChain, vector DB, RAG, reranking, MCP in one working project

**Time:** 8-12 hours (1-2 days)

---

## Why This Project

**Fills these gaps:**
- ✅ LangChain experience (main framework)
- ✅ Vector databases (ChromaDB)
- ✅ RAG architecture (retrieval + generation)
- ✅ Reranking (basic implementation)
- ✅ MCP server (Anthropic's protocol)
- ✅ Performance optimization (caching, streaming)

**Maps to Aircall use case:**
- They need agents to query call data
- Real-time context retrieval
- Integration with external tools (CRM)

---

## Project Architecture

```
┌─────────────────┐
│  Fake Call Data │
│  (JSON)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChromaDB       │
│  Vector Store   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LangChain      │
│  RAG Agent      │
│  - Retriever    │
│  - Reranker     │
│  - Generator    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MCP Server     │
│  (Exposes tools)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MCP Client     │
│  (Claude/GPT)   │
└─────────────────┘
```

---

## Phase 1: RAG with LangChain (6 hours)

### Step 1: Setup (30 min)

```bash
mkdir aircall-context-agent
cd aircall-context-agent
python -m venv venv
source venv/bin/activate
pip install langchain-anthropic langchain-community sentence-transformers chromadb python-dotenv
```

Create `.env`:
```
OPENAI_API_KEY=your-key-here
```

### Step 2: Generate Fake Call Data (30 min)

Create `data/calls.json`:

```json
[
  {
    "call_id": "call_001",
    "from": "+1-555-0100",
    "to": "+1-555-0200",
    "duration": 180,
    "timestamp": "2026-02-01T10:30:00Z",
    "transcript": "Customer called about billing issue. Resolved by adjusting invoice.",
    "tags": ["billing", "resolved"],
    "agent": "Sarah"
  },
  {
    "call_id": "call_002",
    "from": "+1-555-0101",
    "to": "+1-555-0200",
    "duration": 240,
    "timestamp": "2026-02-01T11:00:00Z",
    "transcript": "Technical support call. Customer experiencing call quality issues. Suggested network troubleshooting steps.",
    "tags": ["technical", "call-quality"],
    "agent": "Mike"
  }
  // ... add 10-15 more calls
]
```

**Script to generate more calls:**

```python
# generate_calls.py
import json
import random
from datetime import datetime, timedelta

topics = [
    "billing issue",
    "call quality problem",
    "integration setup",
    "account upgrade",
    "feature request",
]

def generate_call(call_id):
    topic = random.choice(topics)
    return {
        "call_id": f"call_{call_id:03d}",
        "from": f"+1-555-{random.randint(100, 999):04d}",
        "to": "+1-555-0200",
        "duration": random.randint(60, 300),
        "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
        "transcript": f"Customer called about {topic}. Discussed options and resolved.",
        "tags": topic.split(),
        "agent": random.choice(["Sarah", "Mike", "Alex", "Jordan"])
    }

calls = [generate_call(i) for i in range(1, 21)]
with open("data/calls.json", "w") as f:
    json.dump(calls, f, indent=2)
```

### Step 3: Build Vector Store (1 hour)

```python
# ingest.py
import json
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load calls
with open("data/calls.json") as f:
    calls = json.load(f)

# Convert to LangChain Documents
documents = []
for call in calls:
    content = f"""
Call ID: {call['call_id']}
From: {call['from']}
To: {call['to']}
Duration: {call['duration']}s
Timestamp: {call['timestamp']}
Agent: {call['agent']}
Transcript: {call['transcript']}
Tags: {', '.join(call['tags'])}
"""
    doc = Document(
        page_content=content,
        metadata={
            "call_id": call['call_id'],
            "agent": call['agent'],
            "tags": call['tags']
        }
    )
    documents.append(doc)

# Create vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="./chroma_db"
)

print(f"Ingested {len(documents)} calls into ChromaDB")
```

Run: `python ingest.py`

### Step 4: Build RAG Chain (2 hours)

```python
# agent.py
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve top 5
)

# Custom prompt
prompt_template = """You are an AI assistant helping analyze Aircall call data.

Use the following call records to answer the question:

{context}

Question: {question}

Answer the question based on the call data above. If you don't have enough information, say so.
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create chain
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# Test queries
queries = [
    "What billing issues have customers reported?",
    "Which agent handled the most calls?",
    "What call quality problems were mentioned?"
]

for query in queries:
    print(f"\nQ: {query}")
    result = qa_chain.run(query)
    print(f"A: {result}")
```

Run: `python agent.py`

### Step 5: Add Reranking (1.5 hours)

```python
# reranker.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_anthropic import ChatAnthropic

# Create base retriever (same as before)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Add LLM-based reranker
llm = ChatAnthropic(temperature=0, model="claude-3-haiku-20240307")  # Cheaper model for reranking
compressor = LLMChainExtractor.from_llm(llm)

# Compression retriever = retrieval + reranking
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Update QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    chain_type="stuff",
    retriever=compression_retriever,  # Use reranking retriever
    chain_type_kwargs={"prompt": PROMPT}
)
```

**What this does:**
1. Retrieve top 10 from vector DB
2. LLM reranks by relevance to query
3. Keep only most relevant chunks
4. Generate answer

### Step 6: Add Performance Optimization (1 hour)

```python
# optimized_agent.py
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.callbacks import StreamingStdOutCallbackHandler

# Enable caching (avoid re-embedding same queries)
set_llm_cache(InMemoryCache())

# Enable streaming (show results as they come)
llm = ChatAnthropic(
    model="gpt-4",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Use cache for embeddings
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed(text: str):
    return embeddings.embed_query(text)
```

---

## Phase 2: MCP Server (4-6 hours)

### Step 1: Understand MCP (1 hour)

Read: https://modelcontextprotocol.io/docs

**Key concepts:**
- MCP Server = exposes tools/resources to LLMs
- Tools = functions LLM can call
- Resources = data sources LLM can read

### Step 2: Install MCP SDK (30 min)

```bash
pip install mcp
```

### Step 3: Build MCP Server (2 hours)

```python
# mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncio
from agent import qa_chain  # Import your RAG chain

# Create MCP server
server = Server("aircall-context-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_calls",
            description="Search through Aircall call transcripts and data",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or search query"
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_calls":
        query = arguments["query"]
        result = qa_chain.run(query)
        return [TextContent(type="text", text=result)]

if __name__ == "__main__":
    server.run()
```

Run: `python mcp_server.py`

### Step 4: Test with MCP Client (1 hour)

```python
# mcp_client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:", tools)
            
            # Call search_calls tool
            result = await session.call_tool(
                "search_calls",
                {"query": "What billing issues were reported?"}
            )
            print("Result:", result)

asyncio.run(main())
```

### Step 5: Document It (1 hour)

Create `README.md`:

```markdown
# Aircall Call Context Agent

AI agent for searching and analyzing Aircall call data using RAG, LangChain, and MCP.

## Features

- ✅ RAG (Retrieval-Augmented Generation) with ChromaDB vector store
- ✅ LangChain framework for agent orchestration
- ✅ Reranking for improved retrieval accuracy
- ✅ Performance optimization (caching, streaming)
- ✅ MCP Server exposing call search tools
- ✅ Fake call data for demonstration

## Architecture

[Diagram showing data flow]

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_calls.py
python ingest.py
python agent.py
```

## MCP Server

Start server:
```bash
python mcp_server.py
```

Test with client:
```bash
python mcp_client.py
```

## Example Queries

- "What billing issues have customers reported?"
- "Which agent handled the most calls?"
- "What call quality problems were mentioned?"

## Tech Stack

- LangChain 0.1.x
- ChromaDB 0.4.x
- Anthropic API (Claude 3 Sonnet/Haiku)
- HuggingFace (Local Embeddings)
- MCP SDK
- Python 3.11+
```

---

## Deliverables

After 8-12 hours, you'll have:

1. **GitHub repo** with working code
2. **README** explaining architecture
3. **Demo** you can run in interview
4. **Experience** with all required technologies

---

## Interview Talking Points

**On LangChain:**
> "I built this demo project where I used LangChain to orchestrate a RAG pipeline. I chose LangChain because it abstracts away boilerplate for retrieval, reranking, and prompt management. Before this, I'd built agents from scratch, so I understand what's happening under the hood."

**On RAG:**
> "My demo retrieves call transcripts from ChromaDB using semantic search, then reranks the top results with an LLM-based compressor before generating the final answer. This approach improved accuracy vs. just vector similarity."

**On MCP:**
> "I implemented an MCP server that exposes call search as a tool. This maps directly to Aircall's need to give AI agents access to customer data. The MCP protocol standardizes how LLMs interact with external systems."

**On Reranking:**
> "I added a reranking step using LangChain's ContextualCompressionRetriever. It retrieves top 10 from the vector DB, then an LLM scores each by actual relevance. This catches edge cases where semantic similarity doesn't match user intent."

**On Performance Optimization:**
> "I implemented caching for embeddings and LLM responses, plus streaming to reduce perceived latency. In production, I'd also batch embed calls and use a cheaper model for reranking."

---

## Timeline

**Today (Mon Feb 11):** 
- Afternoon: Phase 1 (Steps 1-3) - 3 hours
- Evening: Phase 1 (Steps 4-6) - 3 hours

**Tomorrow (Tue Feb 12):**
- Morning: Phase 2 (MCP) - 3 hours
- Afternoon: Documentation + testing - 2 hours

**Wednesday (Feb 13):**
- Review + prepare talking points
- Deploy to GitHub
- Test run through demo

**Thursday (Feb 13):**
- Interview with Deepanshu
- Be ready to demo if asked

---
