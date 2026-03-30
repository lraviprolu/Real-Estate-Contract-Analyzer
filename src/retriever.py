"""
retriever.py — Hybrid retrieval pipeline:
  1. Dense  : ChromaDB cosine similarity (semantic search)
  2. Sparse : BM25 (exact legal-term matching)
  3. Merge  : Reciprocal Rank Fusion
  4. Rerank : Cross-encoder (ms-marco-MiniLM)
"""

import re
import sys
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    BM25_INDEX_PATH,
    RERANKER_MODEL,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RERANKER_TOP_K,
    RRF_K,
    CONFIDENCE_THRESHOLD,
)
from ingestion import load_bm25_corpus

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_collection = None
_reranker: CrossEncoder | None = None
_bm25: BM25Okapi | None = None
_bm25_texts: list[str] = []
_bm25_metas: list[dict] = []


def _get_collection():
    global _collection
    if _collection is None:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _get_bm25() -> tuple[BM25Okapi | None, list[str], list[dict]]:
    global _bm25, _bm25_texts, _bm25_metas
    if _bm25 is None:
        tokenized, texts, metas = load_bm25_corpus()
        if tokenized:
            _bm25 = BM25Okapi(tokenized)
            _bm25_texts = texts
            _bm25_metas = metas
    return _bm25, _bm25_texts, _bm25_metas


def reload_bm25() -> None:
    """Force-reload BM25 index after new documents are ingested."""
    global _bm25
    _bm25 = None
    _get_bm25()


# ── Dense retrieval ────────────────────────────────────────────────────────────

def dense_search(query: str, top_k: int = DENSE_TOP_K) -> list[dict]:
    """ChromaDB semantic search. Returns chunks above CONFIDENCE_THRESHOLD."""
    collection = _get_collection()
    n_docs = collection.count()
    if n_docs == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, n_docs),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - dist  # cosine distance → similarity
        if score >= CONFIDENCE_THRESHOLD:
            chunks.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "page": meta["page"],
                    "chunk_index": meta["chunk_index"],
                    "dense_score": round(score, 4),
                }
            )
    return chunks


# ── Sparse retrieval ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def sparse_search(query: str, top_k: int = SPARSE_TOP_K) -> list[dict]:
    """BM25 keyword search — excels at exact legal terms like 'contingency', 'escrow'."""
    bm25, texts, metas = _get_bm25()
    if bm25 is None or not texts:
        return []

    scores = bm25.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "text": texts[idx],
            "source": metas[idx]["source"],
            "page": metas[idx]["page"],
            "chunk_index": metas[idx]["chunk_index"],
            "sparse_score": round(float(scores[idx]), 4),
        }
        for idx in top_indices
        if scores[idx] > 0
    ]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _chunk_key(chunk: dict) -> tuple:
    return (chunk["source"], chunk["page"], chunk["chunk_index"])


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """Merge ranked lists from dense and sparse using RRF scoring."""
    rrf_scores: dict[tuple, float] = {}
    chunks_by_key: dict[tuple, dict] = {}

    for rank, chunk in enumerate(dense_results):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunks_by_key[key] = chunk.copy()

    for rank, chunk in enumerate(sparse_results):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key in chunks_by_key:
            chunks_by_key[key]["sparse_score"] = chunk.get("sparse_score")
        else:
            chunks_by_key[key] = chunk.copy()

    merged = []
    for key, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        merged.append({**chunks_by_key[key], "rrf_score": round(score, 6)})

    return merged


# ── Cross-encoder reranker ─────────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_k: int = RERANKER_TOP_K) -> list[dict]:
    """Score each (query, passage) pair with a cross-encoder; return top_k."""
    if not candidates:
        return []

    reranker = _get_reranker()
    ce_scores = reranker.predict([(query, c["text"]) for c in candidates]).tolist()

    for chunk, score in zip(candidates, ce_scores):
        chunk["rerank_score"] = round(float(score), 4)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# ── Full hybrid pipeline ───────────────────────────────────────────────────────

def retrieve(query: str) -> dict[str, Any]:
    """
    Run the full hybrid pipeline.

    Returns:
        dense_results   – ChromaDB top-K (semantic)
        sparse_results  – BM25 top-K (keyword)
        merged_results  – RRF-merged list
        final_chunks    – cross-encoder reranked top-K sent to the LLM
    """
    dense = dense_search(query, top_k=DENSE_TOP_K)
    sparse = sparse_search(query, top_k=SPARSE_TOP_K)
    merged = reciprocal_rank_fusion(dense, sparse)
    final = rerank(query, merged, top_k=RERANKER_TOP_K)

    return {
        "dense_results": dense,
        "sparse_results": sparse,
        "merged_results": merged,
        "final_chunks": final,
    }
