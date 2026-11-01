"""Reading money figures from text, and checking answer figures against the evidence.

Shared by the agent (figure check at answer time) and the evaluation (answer scoring).

Figures are read the way people write them in both languages: "TZS 980,500 milioni",
"shilingi bilioni 980.5" (Swahili puts the scale first), "980.5 billion" and "TSh980.5bn" are
the same amount. Fiscal years such as "2025/26" and citation ids such as [S12] are not amounts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

SCALES = {
    "thousand": 1e3, "elfu": 1e3, "k": 1e3,
    "million": 1e6, "millions": 1e6, "milioni": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "billions": 1e9, "bilioni": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "trillions": 1e12, "trilioni": 1e12, "tn": 1e12,
}
_SCALE = "|".join(sorted(SCALES, key=len, reverse=True))
# Before a number only full words count ("bilioni 980.5"); a lone "m" or "b" there is just text.
_PRE_SCALE = "|".join(sorted((s for s in SCALES if len(s) > 2), key=len, reverse=True))
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# 2025/26, 2025-26, 2025/2026 and FY2025/26 are fiscal years, not amounts.
FISCAL_YEAR = re.compile(r"(?:\bFY\s*)?(?<!\d)(?:19|20)\d{2}\s*[/-]\s*\d{2,4}(?!\d)",
                         re.IGNORECASE)
# Numbers that label something rather than measure it: "Section 117", "Fungu 52", "p. 3".
LABEL_BEFORE = re.compile(
    r"(?:section|sections|sehemu|kifungu|vifungu|vote|votes|fungu|mafungu|page|pages|ukurasa|"
    r"kurasa|item|code|namba|no\.|p\.|pp\.|part|chapter|sura|article|ibara)\s*$",
    re.IGNORECASE)
PERCENT_AFTER = re.compile(r"^\s*(?:%|percent\b|per\s+cent\b)", re.IGNORECASE)
PERCENT_BEFORE = re.compile(r"asilimia\s*$", re.IGNORECASE)
CURRENCY_GLUED = re.compile(r"\b(TZS|TShs?|Tshs?|Shs?|Sh)(?=\d)", re.IGNORECASE)
AMOUNT = re.compile(
    rf"(?<![\w.,])(?:\b(?P<pre>{_PRE_SCALE})\s+)?(?P<num>{_NUM})(?:\s*(?P<post>{_SCALE})\b)?",
    re.IGNORECASE,
)
REL_TOL = 0.005
MIN_CHECKED = 10        # below this, figures are counts or list numbers, not amounts


def _scan(text: str) -> list[tuple[str, float, bool]]:
    """(written form, value, is_amount) for every number; is_amount is False for percentages
    and for labels such as "Section 117"."""
    text = FISCAL_YEAR.sub(" ", text)
    text = CURRENCY_GLUED.sub(r"\1 ", text)    # "Sh57.04" -> "Sh 57.04" (but "S12" stays)
    out = []
    for m in AMOUNT.finditer(text):
        value = float(m.group("num").replace(",", ""))
        scale = m.group("post") or m.group("pre")
        before, after = text[max(0, m.start() - 20):m.start()], text[m.end():m.end() + 12]
        is_amount = not (PERCENT_AFTER.match(after) or PERCENT_BEFORE.search(before)
                         or (not m.group("pre") and LABEL_BEFORE.search(before)))
        out.append((m.group(0).strip(), value * SCALES[scale.lower()] if scale else value,
                    is_amount))
    return out


def extract_amount_spans(text: str) -> list[tuple[str, float]]:
    """(text as written, value) for every number in `text`, percentages included."""
    return [(w, v) for w, v, _ in _scan(text)]


def extract_amounts(text: str) -> list[float]:
    return [v for _, v in extract_amount_spans(text)]


def number_found(expected: float, found: Iterable[float], rel_tol: float = REL_TOL) -> bool:
    return any(math.isclose(expected, f, rel_tol=rel_tol, abs_tol=0.5) for f in found)


def _matches_any_scale(value: float, evidence: list[float]) -> bool:
    """True if value equals an evidence figure at some power of 1000.

    Tables print "980,500" under a "TZS milioni" header; an answer saying "980.5 bilioni" is the
    same amount. Allowing every scale is lenient by design: this check flags figures that appear
    nowhere in the evidence, it does not prove a figure's unit is right.
    """
    for k in range(-4, 5):
        if number_found(value, [e * 1000.0 ** k for e in evidence]):
            return True
    return False


def unverified_figures(answer: str, evidence_texts: Iterable[str]) -> list[str]:
    """Money figures in the answer that appear in none of the evidence texts, at any scale.

    Percentages and labels ("Section 117") are not checked: percentages are usually computed,
    and labels are not amounts.
    """
    evidence = [v for t in evidence_texts for v in extract_amounts(t)]
    flagged = []
    for written, value, is_amount in _scan(answer):
        if not is_amount or abs(value) < MIN_CHECKED or _matches_any_scale(value, evidence):
            continue
        if written not in flagged:
            flagged.append(written)
    return flagged
