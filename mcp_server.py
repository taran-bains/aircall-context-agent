"""MCP Server exposing call search functionality."""
import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize server
server = Server("aircall-context-server")

# Global QA chain (initialized on first use)
_qa_chain = None


def get_qa_chain():
    """Lazy initialization of QA chain."""
    global _qa_chain
    if _qa_chain is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        prompt_template = """Use these call records to answer:

{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
        
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    return _qa_chain


@server.list_tools()
async def list_tools():
    """List available MCP tools."""
    return [
        Tool(
            name="search_calls",
            description="Search through Aircall call transcripts and metadata to answer questions about customer interactions, issues, agents, and trends.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or search query about call data"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_call_stats",
            description="Get statistics about calls (total, by agent, by category, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "stat_type": {
                        "type": "string",
                        "description": "Type of statistic: 'total', 'by_agent', 'by_category'",
                        "enum": ["total", "by_agent", "by_category"]
                    }
                },
                "required": ["stat_type"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle MCP tool calls."""
    if name == "search_calls":
        query = arguments["query"]
        qa_chain = get_qa_chain()
        
        # Run query
        result = qa_chain.invoke({"query": query})
        
        # Format response
        answer = result['result']
        sources = [doc.metadata['call_id'] for doc in result['source_documents'][:3]]
        
        response = f"{answer}\n\nSources: {', '.join(sources)}"
        
        return [TextContent(type="text", text=response)]
    
    elif name == "get_call_stats":
        # Load calls directly from JSON for stats
        with open("data/calls.json") as f:
            calls = json.load(f)
        
        stat_type = arguments["stat_type"]
        
        if stat_type == "total":
            stats = f"Total calls: {len(calls)}"
        
        elif stat_type == "by_agent":
            agent_counts = {}
            for call in calls:
                agent = call['agent']
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            stats = "Calls by agent:\n" + "\n".join(
                f"  {agent}: {count}" for agent, count in sorted(agent_counts.items())
            )
        
        elif stat_type == "by_category":
            category_counts = {}
            for call in calls:
                for tag in call['tags']:
                    category_counts[tag] = category_counts.get(tag, 0) + 1
            stats = "Calls by category:\n" + "\n".join(
                f"  {cat}: {count}" for cat, count in sorted(category_counts.items())
            )
        
        return [TextContent(type="text", text=stats)]


if __name__ == "__main__":
    print("🚀 Starting MCP server: aircall-context-server")
    print("   Tools: search_calls, get_call_stats")
    server.run()
