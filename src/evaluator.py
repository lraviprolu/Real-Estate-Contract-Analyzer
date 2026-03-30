"""
evaluator.py — Faithfulness evaluation via LLM-as-judge.

Sends the retrieved context + question + generated answer to the LLM and asks it
to rate how well the answer is supported by the source clauses (0.0 – 1.0).
"""

import json
import re
import sys
from pathlib import Path

from groq import Groq

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


EVAL_PROMPT = """\
You are evaluating whether an AI-generated answer about a real estate contract \
is faithful to (i.e., supported by) the provided source clauses.

SOURCE CLAUSES:
{context}

QUESTION: {question}

AI ANSWER:
{answer}

Rate faithfulness 0.0 – 1.0:
  1.0 — Every claim is directly supported by the source clauses
  0.7 — Most claims supported; minor extrapolation present
  0.5 — Some claims supported, others not verifiable from context
  0.3 — Few claims supported; significant unsupported content
  0.0 — Answer contradicts or ignores the source clauses entirely

Respond with ONLY valid JSON — no extra text:
{{"score": <float>, "reasoning": "<one sentence>", "unsupported_claims": ["<claim>", ...]}}\
"""


def evaluate_faithfulness(query: str, answer: str, chunks: list[dict]) -> dict:
    """
    Score how faithfully the answer is grounded in retrieved chunks.

    Returns:
        score              – float 0.0–1.0
        reasoning          – one-sentence explanation
        unsupported_claims – list of claims not found in context
    """
    if not chunks or not answer:
        return {
            "score": 0.0,
            "reasoning": "No context or answer to evaluate.",
            "unsupported_claims": [],
        }

    context = "\n\n---\n\n".join(
        f"[{c.get('source', '?')} — Page {c.get('page', '?')}]\n{c['text']}"
        for c in chunks
    )

    prompt = EVAL_PROMPT.format(context=context, question=query, answer=answer)

    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Robust JSON extraction
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {
                "score": min(float(result.get("score", 0.5)), 1.0),
                "reasoning": result.get("reasoning", ""),
                "unsupported_claims": result.get("unsupported_claims", []),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: pull the first float found
    score_match = re.search(r"(\d+\.\d+|\d+)", raw)
    score = min(float(score_match.group()), 1.0) if score_match else 0.5
    return {"score": score, "reasoning": raw[:200], "unsupported_claims": []}
