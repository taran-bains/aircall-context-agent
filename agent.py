"""Basic RAG agent using LangChain and ChromaDB."""
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_qa_chain():
    """Create the RAG QA chain."""
    # Load vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # Retrieve top 10 documents
    )

    # Custom prompt template
    prompt_template = """You are an AI assistant helping analyze Aircall call data.

IMPORTANT LIMITATIONS:
- You are provided with the top 10 most relevant call records based on semantic similarity to the query
- For counting/aggregation queries (e.g., "which agent handled the most calls"), you can only see these 10 records, not the entire dataset
- If a query requires seeing all records for accurate counting, acknowledge this limitation
- For semantic search queries (e.g., "what billing issues were reported"), the retrieval system should find the most relevant records

Use the following call records to answer the question. Be specific and cite call IDs when relevant.

Call Records:
{context}

Question: {question}

Answer based on the call data above. If you don't have enough information or if the query requires full dataset access for accurate results, say so clearly.

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create LLM
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


def main():
    """Run sample queries."""
    print("🤖 Initializing RAG agent...")
    qa_chain = create_qa_chain()
    print("✅ Agent ready\n")

    # Standard queries for comparison across agents
    queries = [
        "What billing issues have customers reported?",
        "What are the main customer pain points?",
        "Which calls were marked as unresolved?",
        "What technical issues have been reported recently?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"{'='*60}")

        result = qa_chain.invoke({"query": query})

        print(f"\nA: {result['result']}")
        print(f"\n📄 Sources:")
        for i, doc in enumerate(result['source_documents'][:10], 1):
            print(f"   {i}. {doc.metadata['call_id']} (Agent: {doc.metadata['agent']})")


if __name__ == "__main__":
    main()
