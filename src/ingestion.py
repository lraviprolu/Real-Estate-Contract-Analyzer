"""
ingestion.py — PDF loading, chunking, and dual indexing:
  Dense : ChromaDB with BGE embeddings
  Sparse: BM25 (rank-bm25) serialized to disk
"""

import re
import pickle
import sys
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CONTRACTS_DIR,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    BM25_INDEX_PATH,
)

# ── ChromaDB ───────────────────────────────────────────────────────────────────

_embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ── PDF → pages ────────────────────────────────────────────────────────────────

def load_pdf(path: Path) -> list[dict]:
    """Return one dict per page: {text, page, source}."""
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"text": text, "page": i + 1, "source": path.name})
    doc.close()
    return pages


# ── Chunking ───────────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Paragraph-aware sliding-window chunker."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_chars:
                # Force-split oversized paragraph with overlap
                for start in range(0, len(para), chunk_chars - overlap_chars):
                    piece = para[start : start + chunk_chars]
                    if len(piece) >= MIN_CHUNK_LENGTH:
                        chunks.append(piece)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) >= MIN_CHUNK_LENGTH]


def chunk_pages(pages: list[dict]) -> list[dict]:
    chunk_chars = CHUNK_SIZE * 4      # ~4 chars per token
    overlap_chars = CHUNK_OVERLAP * 4

    chunks = []
    for page in pages:
        for j, text in enumerate(_split_text(page["text"], chunk_chars, overlap_chars)):
            chunks.append(
                {
                    "text": text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": j,
                }
            )
    return chunks


# ── Dense store (ChromaDB) ─────────────────────────────────────────────────────

def store_chunks_dense(chunks: list[dict], collection) -> None:
    existing_ids = set(collection.get()["ids"])

    new = [
        (f"{c['source']}_p{c['page']}_c{c['chunk_index']}", c)
        for c in chunks
        if f"{c['source']}_p{c['page']}_c{c['chunk_index']}" not in existing_ids
    ]

    if not new:
        return

    collection.add(
        ids=[cid for cid, _ in new],
        documents=[c["text"] for _, c in new],
        metadatas=[
            {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]}
            for _, c in new
        ],
    )


# ── Sparse store (BM25 corpus) ─────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def load_bm25_corpus() -> tuple[list[list[str]], list[str], list[dict]]:
    """Returns (tokenized_corpus, raw_texts, metadata_list)."""
    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return [], [], []


def save_bm25_corpus(
    tokenized: list[list[str]], texts: list[str], metas: list[dict]
) -> None:
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump((tokenized, texts, metas), f)


def store_chunks_sparse(chunks: list[dict]) -> None:
    tokenized, texts, metas = load_bm25_corpus()
    existing_keys = {(m["source"], m["page"], m["chunk_index"]) for m in metas}

    for c in chunks:
        key = (c["source"], c["page"], c["chunk_index"])
        if key not in existing_keys:
            tokenized.append(tokenize(c["text"]))
            texts.append(c["text"])
            metas.append(
                {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]}
            )
            existing_keys.add(key)

    save_bm25_corpus(tokenized, texts, metas)


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_file(path: Path, collection=None) -> int:
    """Ingest a single PDF into both indexes. Returns chunk count."""
    if collection is None:
        collection = get_collection()
    pages = load_pdf(path)
    chunks = chunk_pages(pages)
    store_chunks_dense(chunks, collection)
    store_chunks_sparse(chunks)
    return len(chunks)


def ingest_directory(dir_path: Path = CONTRACTS_DIR, collection=None) -> dict[str, int]:
    """Ingest all PDFs in a directory. Returns {filename: chunk_count}."""
    if collection is None:
        collection = get_collection()
    results = {}
    for pdf in sorted(dir_path.glob("*.pdf")):
        count = ingest_file(pdf, collection)
        results[pdf.name] = count
        print(f"  Ingested {pdf.name}: {count} chunks")
    return results


if __name__ == "__main__":
    print("Ingesting sample contracts...")
    results = ingest_directory()
    total = sum(results.values())
    print(f"Done. {len(results)} files, {total} total chunks.")
