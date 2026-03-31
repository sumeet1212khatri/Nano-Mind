# ============================================================
# test_nanomind.py — NanoMind Unit Tests
# Run: pytest test_nanomind.py -v
# ============================================================

import struct
import tiktoken
import pytest

enc = tiktoken.get_encoding("gpt2")

# ── Tokenizer tests ──────────────────────────────────────────

def test_tokenizer_roundtrip_simple():
    text = "Hello, how are you?"
    ids  = enc.encode_ordinary(text)
    back = enc.decode(ids)
    assert back == text

def test_tokenizer_roundtrip_multiline():
    text = "User: What is AI?\nAssistant: AI is artificial intelligence."
    ids  = enc.encode_ordinary(text)
    back = enc.decode(ids)
    assert back == text

def test_tokenizer_roundtrip_system_format():
    text = "System: You are a helpful assistant.\nUser: Hi\nAssistant:"
    ids  = enc.encode_ordinary(text)
    back = enc.decode(ids)
    assert back == text

def test_eot_token():
    # GPT-2 EOT token must be 50256
    assert enc.eot_token == 50256

def test_encode_returns_valid_ids():
    ids = enc.encode_ordinary("What is machine learning?")
    assert len(ids) > 0
    assert all(0 <= i < 50304 for i in ids)

def test_empty_string():
    ids = enc.encode_ordinary("")
    assert ids == []

# ── vocab_size padding test ───────────────────────────────────

def test_vocab_size_padding():
    # 50304 = 50257 padded to nearest multiple of 64
    # Standard GPU memory alignment trick
    assert 50304 % 64 == 0
    assert 50304 >= 50257   # must cover full GPT-2 vocab
    assert 50304 - 50257 == 47  # exact padding amount

# ── model.bin header tests ────────────────────────────────────

def test_model_bin_exists():
    import os
    assert os.path.exists("model.bin"), "model.bin not found in /app"

def test_model_bin_header():
    with open("model.bin", "rb") as f:
        header = struct.unpack('iiiii', f.read(20))
    n_layer, n_head, n_embd, block_size, vocab_size = header
    assert n_layer    == 16,    f"Expected 16 layers, got {n_layer}"
    assert n_head     == 12,    f"Expected 12 heads, got {n_head}"
    assert n_embd     == 768,   f"Expected 768 embd, got {n_embd}"
    assert block_size == 1024,  f"Expected 1024 block_size, got {block_size}"
    assert vocab_size == 50304, f"Expected 50304 vocab, got {vocab_size}"

def test_model_bin_size():
    import os
    size_mb = os.path.getsize("model.bin") / 1e6
    # 152M params × 4 bytes = ~608MB + header overhead → expect 700-800MB
    assert 700 < size_mb < 850, f"model.bin size {size_mb:.1f}MB out of expected range"

# ── tokenizer.bin header tests ────────────────────────────────

def test_tokenizer_bin_exists():
    import os
    assert os.path.exists("tokenizer.bin"), "tokenizer.bin not found in /app"

def test_tokenizer_bin_vocab_size():
    with open("tokenizer.bin", "rb") as f:
        vocab_size = struct.unpack('i', f.read(4))[0]
    assert vocab_size == 50304

def test_tokenizer_bin_readable():
    """Read first 10 token entries and verify they have valid lengths"""
    with open("tokenizer.bin", "rb") as f:
        vocab_size = struct.unpack('i', f.read(4))[0]
        for _ in range(min(10, vocab_size)):
            length = struct.unpack('i', f.read(4))[0]
            assert 0 < length <= 64, f"Token length {length} out of range"
            token_bytes = f.read(length)
            assert len(token_bytes) == length

# ── Config consistency tests ──────────────────────────────────

def test_head_size_divisible():
    n_embd = 768
    n_head = 12
    assert n_embd % n_head == 0
    head_size = n_embd // n_head
    assert head_size == 64

def test_mlp_expansion():
    # MLP hidden dim = 4 × n_embd (standard GPT-2)
    n_embd = 768
    assert 4 * n_embd == 3072