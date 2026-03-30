"""
app.py — Streamlit UI for the Real Estate Contract & Disclosure Analyzer

Run with:
    streamlit run src/app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONTRACTS_DIR, FAITHFULNESS_THRESHOLD
from ingestion import get_collection, ingest_file, ingest_directory
from retriever import retrieve, reload_bm25
from generator import generate_answer
from evaluator import evaluate_faithfulness

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Contract Analyzer",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .score-high { color: #2ecc71; font-weight: 700; font-size: 1.4em; }
    .score-mid  { color: #f39c12; font-weight: 700; font-size: 1.4em; }
    .score-low  { color: #e74c3c; font-weight: 700; font-size: 1.4em; }

    .clause-card {
        background: #1e1e2e;
        border-left: 4px solid #7289da;
        padding: 0.75em 1em;
        border-radius: 6px;
        margin-bottom: 0.6em;
        font-size: 0.85em;
        line-height: 1.5;
    }
    .source-tag {
        color: #7289da;
        font-size: 0.78em;
        font-weight: 700;
        margin-bottom: 0.3em;
    }
    .pipeline-tag {
        font-size: 0.75em;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "history" not in st.session_state:
    st.session_state.history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Contract Library")

    # Auto-ingest bundled sample contracts on first load
    if not st.session_state.ingested:
        with st.spinner("Indexing sample contracts…"):
            collection = get_collection()
            if collection.count() == 0:
                ingest_directory(CONTRACTS_DIR, collection)
            reload_bm25()
            st.session_state.ingested = True

    collection = get_collection()
    st.success(f"{collection.count()} clause chunks indexed")

    st.markdown("**Bundled sample contracts:**")
    for pdf in sorted(CONTRACTS_DIR.glob("*.pdf")):
        short = pdf.stem.replace("_", " ")
        st.markdown(f"- {short}")

    st.divider()

    st.markdown("**Upload your own contract:**")
    uploaded = st.file_uploader(
        "Drop a PDF here", type="pdf", label_visibility="collapsed"
    )
    if uploaded:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / uploaded.name
            dest.write_bytes(uploaded.read())
            with st.spinner(f"Ingesting {uploaded.name}…"):
                count = ingest_file(dest, get_collection())
                reload_bm25()
        st.success(f"Added {count} chunks from **{uploaded.name}**")

    st.divider()

    st.markdown("**Options**")
    show_pipeline = st.toggle("Show retrieval pipeline", value=False)
    show_history = st.toggle("Show query history", value=False)

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🏡 Real Estate Contract & Disclosure Analyzer")
st.caption(
    "Ask questions about any property disclosure or purchase agreement. "
    "Powered by hybrid RAG (dense + sparse) with a faithfulness score on every answer."
)

# ── Sample questions ───────────────────────────────────────────────────────────
SAMPLES = [
    "Are there any lead paint disclosures?",
    "What are the contingency deadlines?",
    "What are the closing costs and escrow terms?",
    "Is there an easement on the property?",
    "What happens if the buyer defaults?",
    "What seller disclosures are required?",
    "What is the loan estimate breakdown?",
    "What are the cash to close amounts?",
]

st.markdown("**Try asking:**")
cols = st.columns(3)
for i, q in enumerate(SAMPLES):
    if cols[i % 3].button(q, use_container_width=True, key=f"sample_{i}"):
        st.session_state.query_input = q

# ── Query box ──────────────────────────────────────────────────────────────────
query = st.text_input(
    "Your question",
    key="query_input",
    placeholder="e.g. What are the seller's disclosure obligations?",
    label_visibility="collapsed",
)

analyze = st.button("Analyze", type="primary", disabled=not query.strip())

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze and query.strip():

    with st.status("Running hybrid retrieval pipeline…", expanded=True) as status:
        st.write("Dense search (ChromaDB)…")
        pipeline = retrieve(query)
        st.write("Sparse search (BM25)…")
        st.write("Reciprocal Rank Fusion + cross-encoder rerank…")
        final_chunks = pipeline["final_chunks"]
        status.update(label=f"Retrieved {len(final_chunks)} clauses", state="complete")

    with st.spinner("Generating answer…"):
        gen = generate_answer(query, final_chunks)

    with st.spinner("Evaluating faithfulness…"):
        ev = evaluate_faithfulness(query, gen["answer"], final_chunks)

    # ── Answer + score ─────────────────────────────────────────────────────────
    st.divider()
    col_ans, col_score = st.columns([3, 1])

    with col_ans:
        st.subheader("Answer")
        st.markdown(gen["answer"])

    with col_score:
        score = ev["score"]
        if score >= FAITHFULNESS_THRESHOLD:
            css, label = "score-high", "High"
        elif score >= 0.4:
            css, label = "score-mid", "Medium"
        else:
            css, label = "score-low", "Low"

        st.markdown("**Faithfulness**")
        st.markdown(
            f'<p class="{css}">{score:.2f} / 1.00</p>'
            f"<small>{label} confidence</small>",
            unsafe_allow_html=True,
        )
        if ev["reasoning"]:
            st.caption(ev["reasoning"])
        if ev.get("unsupported_claims"):
            with st.expander("Unsupported claims detected"):
                for claim in ev["unsupported_claims"]:
                    st.markdown(f"- {claim}")

    # ── Retrieved clauses ──────────────────────────────────────────────────────
    with st.expander(f"Retrieved Clauses ({len(final_chunks)})", expanded=True):
        for i, chunk in enumerate(final_chunks, 1):
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", "?")
            rerank_sc = chunk.get("rerank_score")
            dense_sc = chunk.get("dense_score")
            sparse_sc = chunk.get("sparse_score")

            scores_str = ""
            if rerank_sc is not None:
                scores_str += f" · rerank: {rerank_sc:.3f}"
            if dense_sc is not None:
                scores_str += f" · dense: {dense_sc:.3f}"
            if sparse_sc is not None:
                scores_str += f" · bm25: {sparse_sc:.2f}"

            preview = chunk["text"][:500] + ("…" if len(chunk["text"]) > 500 else "")
            st.markdown(
                f'<p class="source-tag">[{i}] {source} — Page {page}'
                f'<span class="pipeline-tag">{scores_str}</span></p>'
                f'<div class="clause-card">{preview}</div>',
                unsafe_allow_html=True,
            )

    # ── Pipeline details ───────────────────────────────────────────────────────
    if show_pipeline:
        with st.expander("Pipeline Details"):
            tab1, tab2, tab3 = st.tabs(
                ["Dense (ChromaDB)", "Sparse (BM25)", "After RRF Merge"]
            )

            with tab1:
                st.caption(
                    f"{len(pipeline['dense_results'])} results · "
                    "BGE embeddings + cosine similarity"
                )
                for c in pipeline["dense_results"]:
                    st.markdown(
                        f"**{c['source']} p.{c['page']}** "
                        f"— score: `{c.get('dense_score', 0):.3f}`"
                    )
                    st.text(c["text"][:180] + "…")

            with tab2:
                st.caption(
                    f"{len(pipeline['sparse_results'])} results · "
                    "BM25Okapi keyword matching"
                )
                for c in pipeline["sparse_results"]:
                    st.markdown(
                        f"**{c['source']} p.{c['page']}** "
                        f"— BM25: `{c.get('sparse_score', 0):.3f}`"
                    )
                    st.text(c["text"][:180] + "…")

            with tab3:
                st.caption("Merged via Reciprocal Rank Fusion (before cross-encoder rerank)")
                for c in pipeline["merged_results"][:10]:
                    st.markdown(
                        f"**{c['source']} p.{c['page']}** "
                        f"— RRF: `{c.get('rrf_score', 0):.5f}`"
                    )

    # Save to history
    st.session_state.history.insert(
        0,
        {
            "query": query,
            "answer": gen["answer"],
            "score": ev["score"],
            "n_chunks": len(final_chunks),
        },
    )

# ── History ────────────────────────────────────────────────────────────────────
if show_history and st.session_state.history:
    with st.expander(f"Query History ({len(st.session_state.history)})"):
        for item in st.session_state.history:
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(
                f"**A:** {item['answer'][:300]}"
                f"{'…' if len(item['answer']) > 300 else ''}"
            )
            st.caption(
                f"Faithfulness: {item['score']:.2f} · "
                f"Clauses used: {item['n_chunks']}"
            )
            st.divider()
