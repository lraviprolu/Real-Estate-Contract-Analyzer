"""
config.py — Configuration for the Real Estate Contract & Disclosure Analyzer
Pipeline: Dense (ChromaDB) + Sparse (BM25) → RRF → Cross-Encoder Reranker → Groq LLM
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
CONTRACTS_DIR   = DATA_DIR / "sample_contracts"
CHROMA_DIR      = BASE_DIR / "chroma_db"
BM25_INDEX_PATH = BASE_DIR / "bm25_index.pkl"

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 512   # approx tokens per chunk (1 token ≈ 4 chars)
CHUNK_OVERLAP     = 50    # overlap between consecutive chunks (tokens)
MIN_CHUNK_LENGTH  = 100   # discard chunks shorter than this (chars)

# ── Embedding Model ───────────────────────────────────────────────────────────
# BAAI/bge-small-en-v1.5: 384-dim, ~33M params, runs on CPU
# Top of MTEB benchmark for its size class
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "real_estate_contracts"

# ── Retrieval ─────────────────────────────────────────────────────────────────
DENSE_TOP_K          = 10    # ChromaDB candidates before merging
SPARSE_TOP_K         = 10    # BM25 candidates before merging
RERANKER_TOP_K       = 5     # final chunks sent to LLM after reranking
RETRIEVAL_TOP_K      = 5     # backward-compat alias
CONFIDENCE_THRESHOLD = 0.30  # cosine similarity below this → skip dense result
RRF_K                = 60    # Reciprocal Rank Fusion constant

# ── Reranker ──────────────────────────────────────────────────────────────────
# cross-encoder/ms-marco-MiniLM-L-6-v2: ~90MB, strong passage relevance scorer
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL       = "llama-3.1-8b-instant"  # fastest Groq free-tier model
LLM_MAX_TOKENS  = 1024
LLM_TEMPERATURE = 0.0  # deterministic — contracts need precision, not creativity

# ── Evaluation ────────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7  # score below this flags hallucination risk
