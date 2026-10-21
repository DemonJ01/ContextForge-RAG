from typing import List
import logfire

def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed the specified size.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip(): 
            return []
            
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # A single paragraph can exceed the budget (PDF text often has no
            # blank lines). Split it before packing, preserving every character.
            for start in range(0, len(paragraph), chunk_size):
                p = paragraph[start:start + chunk_size]
                separator = "\n\n" if current_chunk else ""
                if len(current_chunk) + len(separator) + len(p) <= chunk_size:
                    current_chunk += separator + p
                    continue
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
