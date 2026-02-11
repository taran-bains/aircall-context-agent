"""Performance-optimized RAG agent with caching and streaming."""
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Enable caching globally
set_llm_cache(InMemoryCache())


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking streaming."""
    
    def __init__(self):
        self.token_count = 0
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Track each token as it arrives (don't print - main code handles that)."""
        self.token_count += 1
        self.tokens.append(token)
    
    def get_stats(self):
        """Get streaming statistics."""
        return {
            "token_count": self.token_count,
            "avg_token_length": sum(len(t) for t in self.tokens) / max(1, len(self.tokens))
        }


def create_optimized_qa_chain():
    """Create optimized QA chain with caching and streaming."""
    # Load vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    # Prompt
    prompt_template = """Analyze these Aircall call records:

IMPORTANT: You are seeing the top 10 most semantically relevant call records. For counting/aggregation queries, acknowledge that you can only see a subset of the data.

{context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # LLM (streaming via callbacks)
    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        temperature=0
    )
    
    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


def run_query_with_streaming(vectorstore, llm, prompt_template, query, streaming_handler, show_typing=False):
    """Run a query with true token-by-token streaming.
    
    This bypasses RetrievalQA to enable real streaming by:
    1. Manually retrieving relevant documents
    2. Formatting the prompt
    3. Streaming directly from the LLM
    
    Args:
        show_typing: If True, adds a small delay between tokens for visual effect
    """
    import time
    from langchain.schema import HumanMessage
    
    start_time = time.time()
    
    # Step 1: Retrieve documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents(query)
    
    # Step 2: Format context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Step 3: Format the prompt
    prompt = prompt_template.format(context=context, question=query)
    
    # Step 4: Stream from LLM
    print("A: ", end="", flush=True)
    
    # Reset handler token count
    streaming_handler.token_count = 0
    streaming_handler.tokens = []
    
    for chunk in llm.stream(
        [HumanMessage(content=prompt)],
        config={"callbacks": [streaming_handler]}
    ):
        print(chunk.content, end="", flush=True)
        
        # Optional: Add tiny delay to make streaming visible
        if show_typing:
            time.sleep(0.01)  # 10ms delay makes streaming visually obvious
    
    elapsed = time.time() - start_time
    
    stats = streaming_handler.get_stats()
    
    print(f"\n\n📄 Sources: {[doc.metadata['call_id'] for doc in docs[:3]]}")
    print(f"⏱️  Time: {elapsed:.2f}s | 🎯 {stats['token_count']} tokens streamed")


def main():
    """Run optimized agent with real token-by-token streaming."""
    import sys
    
    # Check if --slow flag is passed to demonstrate streaming visually
    show_typing_effect = "--slow" in sys.argv
    
    print("🤖 Initializing optimized RAG agent...")
    print("   ✅ Caching enabled")
    print("   ✅ True token-by-token streaming")
    if show_typing_effect:
        print("   ⏱️  Typing effect enabled (--slow mode)")
    print("\n💡 Tip: Run with 'python optimized_agent.py --slow' to see typing effect\n")
    
    # Load components
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        temperature=0
    )
    
    prompt_template = """Analyze these Aircall call records:

IMPORTANT: You are seeing the top 10 most semantically relevant call records. For counting/aggregation queries, acknowledge that you can only see a subset of the data.

{context}

Question: {question}

Answer:"""
    
    # Create streaming callback handler
    streaming_handler = StreamingCallbackHandler()
    
    queries = [
        "What are the main customer pain points?",
        "How many calls were about billing?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Q{i}: {query}")
        print(f"{'='*60}\n")
        
        run_query_with_streaming(
            vectorstore, 
            llm, 
            prompt_template, 
            query, 
            streaming_handler,
            show_typing=show_typing_effect
        )
    
    print("\n\n✅ Streaming demonstration complete!")
    if not show_typing_effect:
        print("   Run with --slow flag to see typing effect: python optimized_agent.py --slow")


if __name__ == "__main__":
    main()
