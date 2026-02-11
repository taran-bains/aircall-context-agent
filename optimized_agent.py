"""Performance-optimized RAG agent with caching and streaming."""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.callbacks import StreamingStdOutCallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Enable caching globally
set_llm_cache(InMemoryCache())


def create_optimized_qa_chain():
    """Create optimized QA chain with caching and streaming."""
    # Load vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Prompt
    prompt_template = """Analyze these Aircall call records:

{context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # LLM with streaming
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        streaming=True,  # Enable streaming
        callbacks=[StreamingStdOutCallbackHandler()]  # Print as tokens arrive
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


def main():
    """Run optimized agent."""
    print("🤖 Initializing optimized RAG agent...")
    print("   ✅ Caching enabled")
    print("   ✅ Streaming enabled\n")
    
    qa_chain = create_optimized_qa_chain()
    
    queries = [
        "What are the main customer pain points?",
        "How many calls were about billing?",  # Will use cache if asked twice
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"{'='*60}\n")
        print("A: ", end="", flush=True)
        
        # Response will stream token-by-token
        result = qa_chain.invoke({"query": query})
        
        print(f"\n\n📄 Sources: {[doc.metadata['call_id'] for doc in result['source_documents'][:3]]}")
    
    print("\n\n💡 Run the same query twice to see caching in action!")


if __name__ == "__main__":
    main()
