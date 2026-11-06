import json
from pathlib import Path
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Paths & Configurations
EVAL_DATASET_FILE = Path("data/evaluation/eval_dataset.json")
QDRANT_PATH = "data/qdrant_db"
COLLECTION_NAME = "sri_lanka_financial_reports_hybrid"

def run_evaluation():
    if not EVAL_DATASET_FILE.exists():
        print(f"Evaluation dataset not found at {EVAL_DATASET_FILE}. Please generate it first.")
        return

    print("Loading evaluation dataset...")
    with open(EVAL_DATASET_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"Loaded {len(eval_data)} test cases. Initializing retrieval and model components...")

    # Initialize Embeddings & Qdrant Client
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
    
    vector_store = QdrantVectorStore.from_existing_collection(
        path=QDRANT_PATH,
        sparse_embedding=sparse_embedding,
        collection_name="sri_lanka_financial_reports_hybrid",
        embedding=embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    # Initialize Local LLM
    llm = ChatOllama(model="llama3.2", temperature=0.1)

    eval_prompt = ChatPromptTemplate.from_template(
        """You are an expert RAG evaluator. Your job is to compare a Generated Answer against a Ground Truth Answer for a given user question.

User Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Evaluate if the Generated Answer is factually consistent and correctly answers the question based on the Ground Truth.
Provide your evaluation strictly in the following JSON format:
{{
  "is_correct": true or false,
  "reason": "Brief explanation of why it is correct or incorrect"
}}
"""
    )

    results = []
    correct_count = 0

    print("\n--- Running Evaluation Pipeline ---")
    for idx, item in enumerate(eval_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth_answer"]
        expected_page = item["page"]
        expected_source = item["source"]

        print(f"\n[Test Case {idx}/{len(eval_data)}]")
        print(f"Q: {question}")

        # 1. Retrieve Context
        retrieved_docs = retriever.invoke(question)
        
        # Check if expected page is in retrieved context
        retrieved_pages = [doc.metadata.get("page") for doc in retrieved_docs]
        retrieved_sources = [doc.metadata.get("source") for doc in retrieved_docs]
        
        page_hit = any(
            (expected_page - 1) <= page <= (expected_page + 1) 
            for page in retrieved_pages
        )

        print(f"Retrieval Check -> Expected Page: {expected_page} | Retrieved Pages: {retrieved_pages} | Hit: {page_hit}")

        if not page_hit:
            print(f"  [DEBUG] Missed expected page {expected_page}. Top 3 retrieved snippets:")
            for i, doc in enumerate(retrieved_docs[:2]):
                print(f"     [{i+1}] Page {doc.metadata.get('page')}: {doc.page_content[:100]}...")

        # 2. Generate Answer using RAG context
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        generation_prompt = f"""Use the following context to answer the question accurately with citations.
Context:
{context_text}

Question: {question}
"""
        gen_response = llm.invoke(generation_prompt)
        generated_answer = gen_response.content.strip()

        # 3. Judge Correctness using LLM-as-a-Judge
        eval_chain_prompt = eval_prompt.format(
            question=question,
            ground_truth=ground_truth,
            generated_answer=generated_answer
        )
        
        eval_result_raw = llm.invoke(eval_chain_prompt).content.strip()
        
        # Clean JSON parsing
        try:
            if "```json" in eval_result_raw:
                eval_result_raw = eval_result_raw.split("```json")[1].split("```")[0].strip()
            elif "```" in eval_result_raw:
                eval_result_raw = eval_result_raw.split("```")[1].split("```")[0].strip()
                
            eval_json = json.loads(eval_result_raw)
            is_correct = eval_json.get("is_correct", False)
            reason = eval_json.get("reason", "")
        except:
            is_correct = False
            reason = "Failed to parse judge output."

        if is_correct:
            correct_count += 1

        print(f"Evaluation -> Correct: {is_correct} | Reason: {reason}")
        
        results.append({
            "question": question,
            "page_hit": page_hit,
            "is_correct": is_correct,
            "reason": reason
        })

    # Summary Report
    accuracy = (correct_count / len(eval_data)) * 100
    retrieval_hits = sum(1 for res in results if res["page_hit"])
    retrieval_hit_rate = (retrieval_hits / len(eval_data)) * 100
    
    print("\n============================== Evaluator Summary ==============================")
    print(f"Total Test Cases Tested: {len(eval_data)}")
    print(f"Retrieval Hit Rate (k=8): {retrieval_hit_rate:.2f}% ({retrieval_hits}/{len(eval_data)} hits)")
    print(f"Generation Accuracy Score: {accuracy:.2f}% ({correct_count}/{len(eval_data)} correct)")
    print("===============================================================================\n")

if __name__ == "__main__":
    run_evaluation()