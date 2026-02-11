"""Ingest call data into ChromaDB vector store."""
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_calls(file_path: str = "data/calls.json") -> list[dict]:
    """Load calls from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def create_documents(calls: list[dict]) -> list[Document]:
    """Convert call records to LangChain Documents."""
    documents = []
    
    for call in calls:
        # Create rich text content
        content = f"""Call ID: {call['call_id']}
From: {call['from']}
To: {call['to']}
Duration: {call['duration']} seconds
Timestamp: {call['timestamp']}
Agent: {call['agent']}
Tags: {', '.join(call['tags'])}
Sentiment: {call['sentiment']}
Resolved: {'Yes' if call['resolved'] else 'No'}

Transcript:
{call['transcript']}
"""
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "call_id": call['call_id'],
                "agent": call['agent'],
                "tags": call['tags'],
                "sentiment": call['sentiment'],
                "resolved": call['resolved'],
                "timestamp": call['timestamp'],
            }
        )
        documents.append(doc)
    
    return documents


def main():
    """Ingest calls into vector store."""
    print("🔄 Loading call data...")
    calls = load_calls()
    print(f"   Loaded {len(calls)} calls")
    
    print("\n📝 Converting to documents...")
    documents = create_documents(calls)
    print(f"   Created {len(documents)} documents")
    
    print("\n🧮 Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    
    # Create vector store with persistence
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"✅ Ingested {len(documents)} documents into ChromaDB")
    print(f"📁 Vector store saved to ./chroma_db/")
    
    # Test a simple query
    print("\n🔍 Testing retrieval...")
    query = "What billing issues were reported?"
    results = vectorstore.similarity_search(query, k=3)
    print(f"   Query: {query}")
    print(f"   Found {len(results)} relevant documents")
    print(f"   Top result: {results[0].metadata['call_id']}")


if __name__ == "__main__":
    main()
