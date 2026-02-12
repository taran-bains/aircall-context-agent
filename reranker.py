"""RAG agent with LLM-based reranking for improved accuracy."""
import warnings
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

# Suppress LangChain deprecation warning for LLMChainExtractor
# The predict_and_parse deprecation is internal to LangChain and doesn't affect functionality
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")


def create_reranking_qa_chain():
    """Create QA chain with reranking."""
    # Load vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Base retriever (get more docs initially)
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15}  # Retrieve top 15 initially for reranking
    )
    
    # Reranker using LLM
    llm_reranker = ChatAnthropic(temperature=0, model="claude-haiku-4-5")  # Cheaper for reranking
    compressor = LLMChainExtractor.from_llm(llm_reranker)
    
    # Compression retriever = retrieval + reranking
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # Prompt template
    prompt_template = """You are an AI assistant analyzing Aircall call data.

IMPORTANT LIMITATIONS:
- You receive the top 10-15 most relevant call records after retrieval and reranking
- For counting queries (e.g., "which agent handled the most calls"), you can only see these records, not the full dataset
- Acknowledge this limitation if a query requires full dataset access for accurate results
- For semantic search queries, the reranking system prioritizes the most relevant records

Use these call records to answer the question:

{context}

Question: {question}

Provide a detailed answer citing specific call IDs and agents when relevant. If the query requires full dataset visibility, note this limitation.

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Main LLM for generation
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    
    # QA chain with reranking retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


def main():
    """Run queries with reranking."""
    print("🤖 Initializing RAG agent with reranking...")
    qa_chain = create_reranking_qa_chain()
    print("✅ Agent ready (with LLM-based reranking)\n")
    
    queries = [
        "What are the most common customer complaints?",
        "Which calls were marked as unresolved?",
        "What technical issues have been reported recently?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"{'='*60}")
        
        result = qa_chain.invoke({"query": query})
        
        print(f"\nA: {result['result']}")
        print(f"\n📄 Reranked Sources (most relevant):")
        for i, doc in enumerate(result['source_documents'][:3], 1):
            print(f"   {i}. {doc.metadata['call_id']} - {doc.metadata['sentiment']}")


if __name__ == "__main__":
    main()
