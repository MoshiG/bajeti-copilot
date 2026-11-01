"""Deterministic metrics.

Answer correctness checks that every expected figure appears (any unit, within 0.5%) and every
expected keyword appears. Figure reading lives in bajeti.numbers, shared with the agent.

Proportions come with Wilson 95% intervals: with ~80 questions, a few points' difference between
two configurations is often noise, and the results table should say so.
"""

from __future__ import annotations

import math

from bajeti.evaluation.gold import Evidence
from bajeti.numbers import extract_amounts, number_found  # noqa: F401  (re-exported)


def score_answer(text: str, numbers: list[float], keywords: list[str]) -> dict[str, bool]:
    found = extract_amounts(text)
    lowered = text.lower()
    numbers_ok = all(number_found(n, found) for n in numbers)
    keywords_ok = all(k.lower() in lowered for k in keywords)
    return {"numbers_ok": numbers_ok, "keywords_ok": keywords_ok,
            "correct": numbers_ok and keywords_ok}


def matches_evidence(doc_id: str, page_start: int | None, page_end: int | None,
                     evidence: list[Evidence]) -> bool:
    if page_start is None:
        return False
    span = set(range(page_start, (page_end or page_start) + 1))
    return any(doc_id == ev.doc_id and span & set(ev.pages) for ev in evidence)


def first_relevant_rank(hits_meta: list[tuple[str, int, int]], evidence: list[Evidence]) -> int:
    """1-based rank of the first hit overlapping gold evidence; 0 if none."""
    for rank, (doc_id, p0, p1) in enumerate(hits_meta, start=1):
        if matches_evidence(doc_id, p0, p1, evidence):
            return rank
    return 0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(flags: list[bool]) -> dict[str, float | int]:
    n, k = len(flags), sum(flags)
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": k / n if n else 0.0, "ci_low": lo, "ci_high": hi}
