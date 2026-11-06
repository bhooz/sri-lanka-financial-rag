import os
from pathlib import Path
import fitz  # PyMuPDF
import json

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def parse_financial_pdf(pdf_path: Path):
    """Parses a Central Bank or financial PDF preserving page-level structure and metadata."""
    doc = fitz.open(pdf_path)
    parsed_documents = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Basic cleaning: skip empty pages or handle running headers if necessary
        cleaned_text = text.strip()
        if not cleaned_text:
            continue
            
        metadata = {
            "source": pdf_path.name,
            "page": page_num + 1,
            "total_pages": len(doc)
        }
        
        parsed_documents.append({
            "content": cleaned_text,
            "metadata": metadata
        })
        
    return parsed_documents

def run_ingestion():
    all_chunks = []
    for pdf_file in RAW_DIR.glob("*.pdf"):
        print(f"Parsing: {pdf_file.name}...")
        docs = parse_financial_pdf(pdf_file)
        all_chunks.extend(docs)
        
    output_file = PROCESSED_DIR / "corpus_processed.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Ingestion complete. Saved {len(all_chunks)} pages to {output_file}")

if __name__ == "__main__":
    run_ingestion()