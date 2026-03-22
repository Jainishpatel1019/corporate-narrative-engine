import os
import sys
import pytest
import tiktoken

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.chunker import SlidingWindowChunker

@pytest.fixture
def chunker():
    return SlidingWindowChunker(chunk_size=150, overlap=25)

def test_empty_string(chunker):
    assert chunker.chunk("") == []

def test_shorter_than_chunk_size(chunker):
    text = "This is a short text with just a few tokens."
    result = chunker.chunk(text)
    
    assert len(result) == 1
    assert result[0]["text"] == text
    assert result[0]["chunk_index"] == 0
    assert result[0]["start_token"] == 0
    assert result[0]["end_token"] > 0
    assert result[0]["end_token"] < 150

def test_exactly_150_tokens(chunker):
    # Create a string that encodes to exactly 150 tokens
    # Using cl100k_base, "word " is usually 1 token
    text = "word " * 150
    
    # Ensure we actually trigger the code path by matching strictly 150 tokens
    token_count = len(tiktoken.get_encoding("cl100k_base").encode(text))
    if token_count > 150:
        # In case the heuristic fails, resize to exact tokens
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)[:150]
        text = tokenizer.decode(tokens)
        
    result = chunker.chunk(text)
    
    assert len(result) == 1
    assert result[0]["chunk_index"] == 0
    assert result[0]["start_token"] == 0
    assert result[0]["end_token"] == 150

def test_chunk_index_and_consistency(chunker):
    text = "repeat " * 500  # Will definitely exceed 150 tokens
    results = chunker.chunk(text)
    
    assert len(results) > 1
    
    for i, chunk in enumerate(results):
        assert chunk["chunk_index"] == i
        assert chunk["end_token"] - chunk["start_token"] <= 150
        assert chunk["start_token"] >= 0

def test_overlap_correctness():
    # Set chunk_size = 10, overlap = 3
    test_chunker = SlidingWindowChunker(chunk_size=10, overlap=3)
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    results = test_chunker.chunk(text)
    
    assert len(results) >= 2
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    for i in range(len(results) - 1):
        chunk_n = results[i]
        chunk_n_plus_1 = results[i+1]
        
        tokens_n = tokenizer.encode(chunk_n["text"])
        tokens_n_plus_1 = tokenizer.encode(chunk_n_plus_1["text"])
        
        # The last 3 tokens of chunk N should equal the first 3 tokens of chunk N+1
        assert tokens_n[-3:] == tokens_n_plus_1[:3]
