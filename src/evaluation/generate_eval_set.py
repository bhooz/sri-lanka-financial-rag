import json
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

PROCESSED_FILE = Path("data/processed/corpus_processed.json")
EVAL_OUTPUT_FILE = Path("data/evaluation/eval_dataset.json")
EVAL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def generate_synthetic_qa():
    print("Loading processed corpus...")
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    # Initialize local LLM (Ollama - llama3.2)
    llm = ChatOllama(model="llama3.2", temperature=0.1)
    
    eval_dataset = []
    
    # Let's pick a sample of pages (e.g., 5 pages) to test synthetic QA generation
    # You can increase or change this slice later as needed
    sample_pages = corpus[10:15] 
    
    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert financial evaluator. Based on the following document excerpt, generate 1 clear, factual question that can be answered strictly using this text, along with its exact ground-truth answer and the source details.

Document Excerpt:
{content}

Source File: {source}
Page Number: {page}

Return the output strictly in the following JSON format:
{{
  "question": "The generated question here",
  "ground_truth_answer": "The accurate answer based only on the excerpt",
  "source": "{source}",
  "page": {page}
}}
"""
    )
    
    print(f"Generating evaluation pairs from {len(sample_pages)} sample pages...")
    
    for item in sample_pages:
        content = item["content"]
        metadata = item["metadata"]
        
        # Skip if content is too short
        if len(content.strip()) < 100:
            continue
            
        formatted_prompt = prompt_template.format(
            content=content[:2000],  # Limit length for prompt
            source=metadata["source"],
            page=metadata["page"]
        )
        
        try:
            response = llm.invoke(formatted_prompt)
            # Clean response text to ensure valid JSON parsing
            text_response = response.content.strip()
            if "```json" in text_response:
                text_response = text_response.split("```json")[1].split("```")[0].strip()
            elif "```" in text_response:
                text_response = text_response.split("```")[1].split("```")[0].strip()
                
            qa_pair = json.loads(text_response)
            eval_dataset.append(qa_pair)
            print(f"Successfully generated QA for: {metadata['source']} (Page {metadata['page']})")
        except Exception as e:
            print(f"Error generating QA for page {metadata['page']}: {e}")
            
    # Save to file
    with open(EVAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\nEvaluation dataset saved to {EVAL_OUTPUT_FILE} with {len(eval_dataset)} questions!")

if __name__ == "__main__":
    generate_synthetic_qa()