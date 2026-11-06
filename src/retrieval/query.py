from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding, SparseTextEmbedding

# Configuration
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "sri_lanka_financial_reports_hybrid"

class HybridRetriever:
    def __init__(self, db_path=QDRANT_PATH, collection_name=COLLECTION_NAME):
        self.client = QdrantClient(path=db_path)
        self.collection_name = collection_name
        
        # Initialize embedding models
        print("Initializing embedding models for hybrid retrieval...")
        self.dense_model = TextEmbedding("BAAI/bge-small-en-v1.5") # Or your chosen dense model
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")

    def search(self, query_text: str, k: int = 8, dense_weight: float = 0.7, sparse_weight: float = 0.3):
        """
        Performs hybrid search with explicit prefetch weights for dense and sparse vectors.
        """
        # 1. Generate dense vector
        dense_vector = list(self.dense_model.embed([query_text]))[0].tolist()

        # 2. Generate sparse vector (BM25)
        sparse_query = list(self.sparse_model.embed([query_text]))[0]
        indices = sparse_query.indices.tolist()
        values = sparse_query.values.tolist()

        # 3. Query Qdrant with weighted prefetch blocks
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Dense semantic search prefetch
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=k * 2,
                    weight=dense_weight
                ),
                # Sparse BM25 keyword search prefetch
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=k * 2,
                    weight=sparse_weight
                ),
            ],
            # Combine rankings using Reciprocal Rank Fusion (RRF) or DBSF
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k
        )

        # 4. Format points into standard documents/dicts for downstream use
        formatted_results = []
        for point in results.points:
            payload = point.payload or {}
            formatted_results.append({
                "content": payload.get("page_content") or payload.get("content", ""),
                "metadata": {
                    "source": payload.get("source"),
                    "page": payload.get("page")
                },
                "score": point.score
            })

        return formatted_results

if __name__ == "__main__":
    retriever = HybridRetriever()
    test_query = "What is the full form of AML?"
    docs = retriever.search(test_query, k=5)
    
    print(f"\nResults for query: '{test_query}'")
    for idx, doc in enumerate(docs, 1):
        print(f"[{idx}] Page: {doc['metadata']['page']} | Score: {doc['score']:.4f}")
        print(f"    Preview: {doc['content'][:120]}...\n")