from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def parse_pdfs():
    pdf_dir = Path("data/raw")
    processed_output = Path("data/processed/corpus_processed.json")
    processed_output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Configure smaller chunk size and overlap to preserve acronym definitions
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # Reduced size to keep acronyms/definitions tightly packed
        chunk_overlap=100,  # Overlap ensures boundary terms are retained across chunks
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    print("Loading and re-chunking PDFs...")
    for pdf_path in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            splits = text_splitter.split_documents([page])
            
            for split in splits:
                all_chunks.append({
                    "content": split.page_content,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page.metadata.get("page", 0) + 1,
                        "total_pages": len(pages)
                    }
                })

    with open(processed_output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Successfully re-processed {len(all_chunks)} chunks into {processed_output}")

if __name__ == "__main__":
    parse_pdfs()