import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

QDRANT_PATH = "data/qdrant_db"
COLLECTION_NAME = "finance_rag_corpus"

def format_docs(docs):
    """Format retrieved documents into a single context string with source citations."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        content = doc.page_content.strip()
        formatted.append(f"[Source {i}: {source}, Page {page}]\n{content}")
    return "\n\n".join(formatted)

def run_rag_pipeline(question: str):
    print(f"\nUser Question: '{question}'")
    print("="*60)
    
    # 1. Initialize Retrieval
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    print(f"Retrieved {len(retrieved_docs)} context chunks from Qdrant.")
    
    # 2. Define Generation Prompt (incorporating citation & refusal rules)
    template = """You are an expert financial and economic AI assistant for Sri Lanka. 
Answer the user's question strictly using only the provided context below. 
If the answer cannot be found in the context, explicitly state "I cannot find the answer in the provided documents." 
Always cite the source document name and page number for your claims using the format [Source X, Page Y].

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # 3. Initialize LLM (Using GPT-4o-mini or GPT-3.5-turbo; ensure OPENAI_API_KEY is set in your environment)
    # Alternatively, you can use ChatOllama(model="llama3") for local execution.
    llm = ChatOllama(model="llama3.2", temperature=0.0)
    
    # 4. Build RAG Chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 5. Invoke Chain
    print("Generating grounded answer with citations...\n")
    answer = rag_chain.invoke(question)
    
    print("--- Final Answer ---")
    print(answer)
    print("="*60)
    return answer

if __name__ == "__main__":
    # Test query against your financial stability / annual economic reports
    sample_question = "What are the key risks to financial stability and banking sector performance?"
    run_g = run_rag_pipeline(sample_question)