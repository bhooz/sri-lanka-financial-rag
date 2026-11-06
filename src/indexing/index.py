import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

PROCESSED_FILE = Path("data/processed/corpus_processed.json")
QDRANT_PATH = "data/qdrant_db"  # Local embedded Qdrant storage folder
COLLECTION_NAME = "finance_rag_corpus"

def run_indexing():
    print("Loading processed corpus...")
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} pages from corpus.")
    
    # Convert JSON pages to LangChain Document objects
    documents = []
    for item in data:
        doc = Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        documents.append(doc)
        
    # Configure text splitter (Chunking strategy experiment baseline)
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs_chunked = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(docs_chunked)} chunks.")
    
    # Initialize Embedding Model (Open-source BGE or MiniLM)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize local Qdrant client
    print(f"Initializing local Qdrant database at '{QDRANT_PATH}'...")
    client = QdrantClient(path=QDRANT_PATH)
    
    # Check if collection exists, if not create it
    collections = client.get_collections().collections
    exists = any(col.name == COLLECTION_NAME for col in collections)
    
    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {COLLECTION_NAME}")
    
    # Store chunks in Qdrant via LangChain wrapper
    print("Embedding and indexing chunks into Qdrant (this may take a moment)...")
    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )
    
    # Add documents in batches to avoid memory spikes
    vector_store.add_documents(docs_chunked)
    print(f"Successfully indexed {len(docs_chunked)} chunks into Qdrant!")

if __name__ == "__main__":
    run_indexing()