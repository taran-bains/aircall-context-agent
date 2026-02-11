"""Basic RAG agent using LangChain and ChromaDB."""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_qa_chain():
    """Create the RAG QA chain."""
    # Load vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 documents
    )
    
    # Custom prompt template
    prompt_template = """You are an AI assistant helping analyze Aircall call data.

Use the following call records to answer the question. Be specific and cite call IDs when relevant.

Call Records:
{context}

Question: {question}

Answer based on the call data above. If you don't have enough information, say so clearly.

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
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
    
    # Sample queries
    queries = [
        "What billing issues have customers reported?",
        "Which agent handled the most calls?",
        "What call quality problems were mentioned?",
        "Show me unresolved issues",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"{'='*60}")
        
        result = qa_chain.invoke({"query": query})
        
        print(f"\nA: {result['result']}")
        print(f"\n📄 Sources:")
        for i, doc in enumerate(result['source_documents'][:3], 1):
            print(f"   {i}. {doc.metadata['call_id']} (Agent: {doc.metadata['agent']})")


if __name__ == "__main__":
    main()
