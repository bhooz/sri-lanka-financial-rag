from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

QDRANT_PATH = "data/qdrant_db"
COLLECTION_NAME = "finance_rag_corpus"

def test_retrieval(query_text: str, k: int = 4):
    print(f"\nQuerying: '{query_text}'\n" + "-"*50)
    
    # Load embeddings and connect to local Qdrant
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = QdrantClient(path=QDRANT_PATH)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    # Perform similarity search
    results = vector_store.similarity_search_with_score(query_text, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        print(f"\n[Result {i}] Score: {score:.4f} | Source: {source} (Page {page})")
        print(f"Content snippet:\n{doc.page_content[:350]}...\n")

if __name__ == "__main__":
    # Test with a realistic financial query tailored to your documents
    sample_query = "What are the key risks to financial stability and banking sector performance?"
    test_retrieval(sample_query, k=3)