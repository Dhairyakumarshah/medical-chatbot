from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file"""
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {page_number + 1} ---\n"
            full_text += text
    
    return full_text

def chunk_by_sentences(text, sentences_per_chunk=5, overlap=2):
    """Split text into chunks by sentences with overlap"""
    # Split into sentences
    sentences = text.replace('\n', ' ').split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    start = 0
    
    while start < len(sentences):
        end = start + sentences_per_chunk
        chunk = ' '.join(sentences[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    
    return chunks

def process_document(pdf_path):
    """Full pipeline: PDF -> text -> chunks"""
    print(f"Processing: {pdf_path}")
    
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    
    chunks = chunk_by_sentences(text)
    print(f"Created {len(chunks)} chunks")
    
    return chunks