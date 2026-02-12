"""Compare Basic RAG vs Reranking on tricky queries."""
from agent import create_qa_chain
from reranker import create_reranking_qa_chain
from dotenv import load_dotenv
import time

load_dotenv()

def compare_agents():
    print("🤖 Initializing agents...")
    basic_chain = create_qa_chain()
    rerank_chain = create_reranking_qa_chain()
    print("✅ Agents ready\n")
    
    # Tricky query designed to fool vector search
    query = "Which customers are explicitly angry or frustrated with the service quality?"
    
    print(f"\n🎯 TEST QUERY: '{query}'")
    print("=" * 80)
    
    # 1. Run Basic Agent
    print("\n🔵 BASIC AGENT (Vector Similarity Only)")
    print("-" * 40)
    start = time.time()
    basic_result = basic_chain.invoke({"query": query})
    print(f"⏱️  Time: {time.time() - start:.2f}s")
    print(f"Answer: {basic_result['result']}")
    print("\nTop Retrieved Docs:")
    for i, doc in enumerate(basic_result['source_documents'][:3], 1):
        content = doc.page_content
        # Robust parsing of transcript
        if "Transcript:" in content:
            transcript = content.split('Transcript:')[1].strip()
        else:
            transcript = content.strip()
        print(f"   {i}. {doc.metadata['call_id']}: {transcript[:100]}...")

    # 2. Run Reranker Agent
    print("\n\n🟣 RERANKER AGENT (Vector + LLM Filtering)")
    print("-" * 40)
    start = time.time()
    rerank_result = rerank_chain.invoke({"query": query})
    print(f"⏱️  Time: {time.time() - start:.2f}s")
    print(f"Answer: {rerank_result['result']}")
    print("\nTop Reranked Docs:")
    for i, doc in enumerate(rerank_result['source_documents'][:3], 1):
        content = doc.page_content
        # Robust parsing of transcript
        if "Transcript:" in content:
            transcript = content.split('Transcript:')[1].strip()
        else:
            transcript = content.strip()
        print(f"   {i}. {doc.metadata['call_id']}: {transcript[:100]}...")

if __name__ == "__main__":
    compare_agents()
