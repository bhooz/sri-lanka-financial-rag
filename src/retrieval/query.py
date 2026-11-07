from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

# Configuration
QDRANT_PATH = "data/qdrant_db"
COLLECTION_NAME = "sri_lanka_financial_reports_hybrid"

class HybridRetriever:
    def __init__(self, db_path=QDRANT_PATH, collection_name=COLLECTION_NAME):
        print("Initializing LangChain Hybrid Retriever...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Connect to existing Qdrant collection using hybrid mode
        self.vector_store = QdrantVectorStore.from_existing_collection(
            path=db_path,
            sparse_embedding=self.sparse_embedding,
            collection_name=collection_name,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )

    def search(self, query_text: str, k: int = 8, **kwargs):
        """
        Performs hybrid search using LangChain QdrantVectorStore retriever.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query_text)
        
        # Format results into standard dictionaries to match evaluation script expectations
        formatted_results = []
        for doc in docs:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1.0  # Default placeholder score for RRF/Hybrid results
            })

        return formatted_results

if __name__ == "__main__":
    retriever = HybridRetriever()
    test_query = "What is the full form of AML?"
    docs = retriever.search(test_query, k=5)
    
    print(f"\nResults for query: '{test_query}'")
    for idx, doc in enumerate(docs, 1):
        print(f"[{idx}] Page: {doc['metadata'].get('page')} | Source: {doc['metadata'].get('source')}")
        print(f"    Preview: {doc['content'][:120]}...\n")