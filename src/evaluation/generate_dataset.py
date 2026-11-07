import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Paths
PDF_DIR = Path("data/raw")  # Path to your source financial PDFs
OUTPUT_DATASET = Path("data/evaluation/eval_dataset.json")

def generate_evaluation_dataset():
    print("Loading financial reports for dataset generation...")
    
    # 1. Load source PDFs (adjust pattern to match your financial report filenames)
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}. Please place your reports there.")
        return

    documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
            documents.append(doc)

    print(f"Loaded {len(documents)} total pages across {len(pdf_files)} documents.")

    # 2. Initialize Local LLM for Synthetic Q&A Generation
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert financial analyst and RAG dataset generator. 
Based on the excerpt from the Sri Lankan financial report below, generate ONE challenging evaluation test case consisting of:
1. A specific user question that can be answered using this text.
2. The exact ground truth answer based only on the text.

Page Content:
{page_content}

Provide your response strictly in the following JSON format:
{{
  "question": "...",
  "ground_truth_answer": "..."
}}
"""
    )

    test_cases = []
    
    # Select a diverse spread of pages (e.g., every Nth page or key sections)
    # Target 25 test cases
    step = max(1, len(documents) // 50) # sample across documents
    sampled_docs = documents[::step] # take a pool to generate 25

    print(f"Generating 25 test cases from sampled document pages...")

    for doc in sampled_docs:
        if len(test_cases) >= 25:
            break
            
        page_num = doc.metadata.get("page", 1) + 1 # 1-based page index
        source_name = doc.metadata.get("source", "financial_report.pdf")
        
        # Skip empty or very short pages (like cover pages)
        if len(doc.page_content.strip()) < 200:
            continue

        chain_prompt = prompt_template.format(page_content=doc.page_content[:1500])
        response = llm.invoke(chain_prompt).content.strip()

        try:
            # Clean JSON formatting
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            data = json.loads(response)
            
            test_cases.append({
                "question": data["question"],
                "ground_truth_answer": data["ground_truth_answer"],
                "page": page_num,
                "source": source_name
            })
            print(f"[{len(test_cases)}/25] Generated test case for page {page_num} of {source_name}")
        except Exception as e:
            # Skip malformed LLM outputs
            continue

    # 3. Save to evaluation dataset file
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=4, ensure_ascii=False)

    print(f"\nSuccessfully generated and saved {len(test_cases)} test cases to {OUTPUT_DATASET}!")

if __name__ == "__main__":
    generate_evaluation_dataset()