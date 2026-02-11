"""MCP Server exposing call search functionality using FastMCP."""
import json
from mcp.server.fastmcp import FastMCP
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("aircall-context-server")

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
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        prompt_template = """Use these call records to answer:

NOTE: You are seeing the top 10 most relevant call records. For queries requiring full dataset visibility (counting, aggregations), note this limitation.

{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
        
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    return _qa_chain


@mcp.tool()
def search_calls(query: str) -> str:
    """Search through Aircall call transcripts and metadata to answer questions about customer interactions, issues, agents, and trends.
    
    Args:
        query: The question or search query about call data
        
    Returns:
        Answer with sources from relevant call records
    """
    qa_chain = get_qa_chain()
    
    # Run query
    result = qa_chain.invoke({"query": query})
    
    # Format response
    answer = result['result']
    sources = [doc.metadata['call_id'] for doc in result['source_documents'][:3]]
    
    response = f"{answer}\n\nSources: {', '.join(sources)}"
    
    return response


@mcp.tool()
def get_call_stats(stat_type: str) -> str:
    """Get statistics about calls (total, by agent, by category, etc.).
    
    Args:
        stat_type: Type of statistic - 'total', 'by_agent', or 'by_category'
        
    Returns:
        Statistics about the calls
    """
    # Load calls directly from JSON for stats
    with open("data/calls.json") as f:
        calls = json.load(f)
    
    if stat_type == "total":
        return f"Total calls: {len(calls)}"
    
    elif stat_type == "by_agent":
        agent_counts = {}
        for call in calls:
            agent = call['agent']
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        stats = "Calls by agent:\n" + "\n".join(
            f"  {agent}: {count}" for agent, count in sorted(agent_counts.items())
        )
        return stats
    
    elif stat_type == "by_category":
        category_counts = {}
        for call in calls:
            for tag in call['tags']:
                category_counts[tag] = category_counts.get(tag, 0) + 1
        stats = "Calls by category:\n" + "\n".join(
            f"  {cat}: {count}" for cat, count in sorted(category_counts.items())
        )
        return stats
    
    else:
        return f"Unknown stat_type: {stat_type}. Use 'total', 'by_agent', or 'by_category'."


if __name__ == "__main__":
    import sys
    # Use stderr for logging - stdout is reserved for JSON-RPC messages
    print("🚀 Starting MCP server: aircall-context-server", file=sys.stderr)
    print("   Tools: search_calls, get_call_stats", file=sys.stderr)
    print("   Transport: stdio", file=sys.stderr)
    mcp.run(transport="stdio")
