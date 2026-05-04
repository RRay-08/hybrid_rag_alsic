import os
import json
import re
from pypdf import PdfReader

# Configuration
PDF_FOLDER = "../data/raw_pdfs"
OUTPUT_FILE = "../data/processed/chunks.json"
CHUNK_SIZE = 512      # tokens per chunk (approx)
CHUNK_OVERLAP = 50    # overlap between chunks

def extract_text_from_pdf(pdf_path):
    """Extract clean text from a single PDF"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        # Basic cleaning
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = re.sub(r'[^\x20-\x7E\n]', '', text)  # Keep only printable ASCII + newlines
        return text.strip()
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")
        return None

def split_into_chunks(text, chunk_size, overlap):
    """Simple chunking by character count (approximate token count)"""
    chunks = []
    start = 0
    # Approximate: 1 token ≈ 4 characters in English technical text
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4
    
    while start < len(text):
        end = start + char_chunk_size
        # Try to break at a sentence boundary
        if end < len(text):
            # Look for period + space in the last 100 chars
            break_point = text.rfind('. ', end-100, end)
            if break_point > start:
                end = break_point + 2
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # Skip tiny chunks
            chunks.append(chunk)
        start = end - char_overlap
        if start >= len(text):
            break
    return chunks

def process_all_pdfs():
    """Main function: process every PDF in the folder"""
    all_chunks = []
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    print(f"📚 Found {len(pdf_files)} PDFs to process")
    
    for i, pdf_name in enumerate(pdf_files, 1):
        pdf_path = os.path.join(PDF_FOLDER, pdf_name)
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue
            
        print(f"   Extracted {len(text)} characters")
        
        # Split into chunks
        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"   Split into {len(chunks)} chunks")
        
        # Store with metadata
        for j, chunk_text in enumerate(chunks):
            all_chunks.append({
                "source_file": pdf_name,
                "chunk_id": f"{pdf_name}_chunk_{j+1}",
                "text": chunk_text,
                "char_length": len(chunk_text)
            })
    
    # Save all chunks to JSON
    print(f"\n💾 Saving {len(all_chunks)} total chunks to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print("✅ process_pdfs.py completed successfully")
    return len(all_chunks)

if __name__ == "__main__":
    process_all_pdfs()