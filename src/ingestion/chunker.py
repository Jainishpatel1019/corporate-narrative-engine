import tiktoken
from typing import List, Dict

class SlidingWindowChunker:
    def __init__(self, chunk_size: int = 150, overlap: int = 25, model: str = "cl100k_base"):
        if overlap >= chunk_size:
            raise ValueError("Overlap cannot be greater than or equal to chunk size.")
            
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding(model)

    def chunk(self, text: str) -> List[Dict]:
        if not text:
            return []
            
        tokens = self.tokenizer.encode(text)
        
        if not tokens:
            return []
            
        token_count = len(tokens)
        
        # Edge case: text shorter than or equal to chunk_size
        if token_count <= self.chunk_size:
            return [{
                "text": text,
                "start_token": 0,
                "end_token": token_count,
                "chunk_index": 0
            }]
            
        chunks = []
        step = self.chunk_size - self.overlap
        chunk_index = 0
        start = 0
        
        while start < token_count:
            end = min(start + self.chunk_size, token_count)
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "start_token": start,
                "end_token": end,
                "chunk_index": chunk_index
            })
            
            chunk_index += 1
            if end == token_count:
                break
                
            start += step
            
        return chunks
