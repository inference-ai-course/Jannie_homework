import fitz  # PyMuPDF
import json
import os
import re
from tqdm import tqdm
from typing import List

PDF_DIR = "data/raw_pdfs"
OUT_CHUNKS = "data/chunks.jsonl"

def clean_text(text: str) -> str:
    """
    Clean and normalize text from PDF extraction.
    """
    # Replace common PDF extraction artifacts
    replacements = {
        '\\ufb01': 'fi',
        '\\ufb02': 'fl',
        '\\u2013': '-',
        '\\u2014': '--',
        '\\u2018': "'",
        '\\u2019': "'",
        '\\u201c': '"',
        '\\u201d': '"',
        '\\ufb00': 'ff',
        '\\ufb03': 'ffi',
        '\\ufb04': 'ffl'
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'-\n', '', text)  # Handle hyphenated line breaks
    text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
    
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_text = page.get_text("text")
        cleaned_text = clean_text(page_text)
        pages.append(cleaned_text)
    full_text = " ".join(pages)
    return full_text

def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks of roughly max_tokens words.
    """
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]
    
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks

def process_pdfs(pdf_dir: str = PDF_DIR) -> List[dict]:
    """
    Process all PDFs in directory and return chunks with metadata.
    """
    all_chunks = []
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, filename)
        paper_id = filename.replace(".pdf", "")
        
        try:
            # Extract and clean text
            text = extract_text_from_pdf(pdf_path)
            
            if not text or len(text.strip()) < 100:  # Skip very short texts
                print(f"Warning: {filename} has very little text, skipping")
                continue
            
            # Chunk text
            text_chunks = chunk_text(text)
            
            # Create chunk objects with metadata
            for i, chunk_content in enumerate(text_chunks):
                chunk_obj = {
                    "id": f"{paper_id}_chunk_{i}",
                    "text": chunk_content,
                    "meta": {
                        "paper_id": paper_id,
                        "chunk_index": i,
                        "filename": filename,
                        "chunk_length": len(chunk_content.split())
                    }
                }
                all_chunks.append(chunk_obj)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Save chunks to JSONL
    with open(OUT_CHUNKS, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    print(f"Successfully processed {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
    return all_chunks

if __name__ == "__main__":
    chunks = process_pdfs()
