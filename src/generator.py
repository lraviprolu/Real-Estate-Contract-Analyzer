"""
generator.py — RAG answer generation via Groq LLM.
Builds a numbered context block from retrieved chunks and enforces
citation-only answers through the system prompt.
"""

import sys
from pathlib import Path

from groq import Groq

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = """\
You are a real estate contract analyst. You answer questions about property \
disclosure documents and purchase agreements based ONLY on the provided contract clauses.

Rules:
- Answer exclusively from the provided context. If the context is insufficient, say so clearly.
- Cite every key claim with its source and page, e.g. [TX_TREC_One_to_Four_Family_Residential.pdf, Page 3].
- After quoting or paraphrasing a legal term, briefly explain it in plain language.
- If multiple documents address the same topic with different terms, note the differences explicitly.
- Never invent clauses, dates, dollar amounts, or deadlines that do not appear in the context.\
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        parts.append(f"[{i}] {source} — Page {page}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate an answer grounded in retrieved chunks.

    Returns:
        answer       – LLM response string
        context_used – formatted context block sent to the LLM
    """
    if not chunks:
        return {
            "answer": (
                "No relevant clauses were found in the loaded contracts "
                "to answer this question."
            ),
            "context_used": "",
        }

    context = build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Contract clauses:\n\n{context}\n\n---\n\nQuestion: {query}",
        },
    ]

    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "context_used": context,
    }
