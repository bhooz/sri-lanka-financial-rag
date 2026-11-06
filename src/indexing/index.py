import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

# Paths & Settings
PROCESSED_DATA_PATH = Path("data/processed/corpus_processed.json")
QDRANT_PATH = "data/qdrant_db"
COLLECTION_NAME = "sri_lanka_financial_reports_hybrid"

def build_hybrid_index():
    print("Loading processed chunks...")
    with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert raw chunks to LangChain Document objects
    documents = [
        Document(
            page_content=item["content"],
            metadata={
                "source": item["metadata"].get("source", ""),
                "page": item["metadata"].get("page", 0)
            }
        )
        for item in data
    ]
    print(f"Loaded {len(documents)} document chunks.")

    # 1. Initialize Dense (Semantic) & Sparse (BM25 Keyword) Embeddings
    print("Initializing embedding models...")
    dense_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # 2. Build or Overwrite Qdrant Collection with Hybrid Retrieval Mode
    print("Indexing documents into Qdrant in HYBRID mode...")
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    print(f"Indexing complete! Collection '{COLLECTION_NAME}' created at {QDRANT_PATH}.")

if __name__ == "__main__":
    build_hybrid_index()