"""LLM judge for faithfulness: does the cited evidence support each claim in the answer?

Evaluation-time only, never in the live demo. For every sentence that carries citations, a
judge model reads the sentence and the passages it cites, and answers (factual-looking sentences
with no citation are labelled "uncited" without a model call, and count as not supported):
    supported    every fact in the sentence (figures, names, years) is in the passages
    partial      some facts are supported, others are not
    unsupported  the passages do not support the sentence

Two choices make the scores credible:
  - use a different, preferably larger, model as judge than the one answering, because a model
    grading its own answers tends to be lenient (configure it under `judge:` in the YAML)
  - validate the judge: `bajeti export-claims` samples claims for you to label by hand, and
    `bajeti judge-validate` reports agreement and Cohen's kappa between you and the judge
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from bajeti.llm import LLMClient

LABELS = ("supported", "partial", "unsupported")
CITE = re.compile(r"\[(S\d+)\]")

JUDGE_SYSTEM = """\
You are a strict fact-checker for answers about Tanzania's public budget documents.
You will see PASSAGES (the evidence an assistant cited) and a CLAIM (one sentence of its answer).
The passages and the claim may be in different languages (Swahili or English); judge meaning.

Label the claim:
- "supported": every fact in the claim (amounts, names, votes, years, measures) is stated in,
  or directly follows from, the passages. Unit conversions (milioni vs bilioni) are fine if the
  value is the same.
- "partial": some facts are supported and at least one is not.
- "unsupported": the passages do not support the claim, or contradict it.

Reply with JSON only: {"label": "supported" | "partial" | "unsupported", "reason": "<short>"}"""


@dataclass
class Claim:
    sentence: str
    source_ids: list[str]
    passages: list[str]


@dataclass
class Verdict:
    label: str            # one of LABELS, or "unparseable"
    reason: str = ""
    raw: str = ""


ABBREVIATIONS = re.compile(r"\b(Dkt|Dk|Mhe|Mh|Bw|Bi|Prof|Dr|Mr|Mrs|Ms|No|Nos|p|pp|Sh|Shs|"
                           r"TShs|e\.g|i\.e|etc|cf|vs)\.", re.IGNORECASE)
MIN_CLAIM_WORDS = 5


def _sentences(text: str) -> list[str]:
    """Sentences and bullet items, without splitting at abbreviations such as 'Dkt.'."""
    protected = ABBREVIATIONS.sub(lambda m: m.group(0)[:-1] + "\x00", text)
    parts = []
    for line in re.split(r"\n+", protected):
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)        # bullet / numbered items
        parts += re.split(r"(?<=[.!?])\s+", line)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def split_claims(answer: str, excerpts: dict[str, str],
                 include_uncited: bool = False) -> list[Claim]:
    """Claims in an answer, each with the passages it cites.

    With include_uncited=True, factual-looking sentences without citations are returned too
    (with no sources), so they count against faithfulness instead of being skipped.
    """
    from bajeti.agent import is_refusal, normalize_citations

    # Move citations written after the full stop ("... 2025/26. [S1]") inside the sentence.
    text = re.sub(r"([.!?])\s*((?:\[S\d+\])+)", r" \2\1", normalize_citations(answer))
    claims = []
    for sentence in _sentences(text):
        ids = [i for i in dict.fromkeys(CITE.findall(sentence)) if i in excerpts]
        clean = re.sub(r"\s*\[S\d+\]", "", sentence).strip()
        if ids:
            claims.append(Claim(clean, ids, [excerpts[i] for i in ids]))
        elif (include_uncited and len(clean.split()) >= MIN_CLAIM_WORDS
              and not clean.endswith(":") and not is_refusal(clean)):
            claims.append(Claim(clean, [], []))
    return claims


def parse_verdict(raw: str) -> Verdict:
    match = re.search(r"\{.*?\}", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            label = str(data.get("label", "")).strip().lower()
            if label in LABELS:
                return Verdict(label, str(data.get("reason", "")), raw)
        except json.JSONDecodeError:
            pass
    words = re.findall(r"\b(unsupported|partial|supported)\b", raw.lower())
    if len(set(words)) == 1:                 # one unambiguous label word
        return Verdict(words[0], "", raw)
    return Verdict("unparseable", "", raw)


class Judge:
    def __init__(self, llm: LLMClient, max_passage_chars: int = 2500) -> None:
        self.llm, self.max_chars = llm, max_passage_chars

    def judge(self, claim: Claim) -> Verdict:
        passages = "\n\n".join(f"[{sid}] {p[: self.max_chars]}"
                               for sid, p in zip(claim.source_ids, claim.passages, strict=True))
        user = f"PASSAGES:\n{passages}\n\nCLAIM:\n{claim.sentence}"
        result = self.llm.chat([{"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user", "content": user}])
        return parse_verdict(result.content or "")

    def judge_answer(self, answer: str, excerpts: dict[str, str]) -> list[dict[str, Any]]:
        out = []
        for claim in split_claims(answer, excerpts, include_uncited=True):
            if not claim.source_ids:      # nothing to check it against: counts as unsupported
                out.append({"sentence": claim.sentence, "sources": [], "label": "uncited",
                            "reason": "no citation"})
                continue
            v = self.judge(claim)
            out.append({"sentence": claim.sentence, "sources": claim.source_ids,
                        "label": v.label, "reason": v.reason})
        return out


# --- validating the judge against human labels -------------------------------------------------


def sample_claims(records: list[dict[str, Any]], n: int, seed: int = 13) -> list[dict[str, Any]]:
    """Claims from evaluation records, for a human to label. Seeded, so the sample is stable."""
    pool = []
    for r in records:
        if not r.get("answerable") or r.get("refused"):
            continue
        excerpts = {c["id"]: c.get("excerpt", "") for c in r.get("citations", [])}
        for i, claim in enumerate(split_claims(r.get("answer", ""), excerpts)):
            pool.append({"claim_id": f"{r['id']}-{i}", "question_id": r["id"],
                         "sentence": claim.sentence, "sources": claim.source_ids,
                         "passages": claim.passages, "human_label": ""})
    random.Random(seed).shuffle(pool)
    return pool[:n]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def validate_judge(judge: Judge, labelled: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in labelled if r.get("human_label") in LABELS]
    if not rows:
        raise ValueError("no claims with human_label set to one of " + ", ".join(LABELS))
    human, model = [], []
    details = []
    for r in rows:
        v = judge.judge(Claim(r["sentence"], r["sources"], r["passages"]))
        human.append(r["human_label"])
        model.append(v.label)
        details.append({"claim_id": r["claim_id"], "human": r["human_label"], "judge": v.label,
                        "reason": v.reason})
    agree = sum(h == m for h, m in zip(human, model, strict=True))
    confusion = {h: {m: sum(1 for x, y in zip(human, model, strict=True) if x == h and y == m)
                     for m in (*LABELS, "unparseable")} for h in LABELS}
    return {"n": len(rows), "agreement": agree / len(rows), "agreed": agree,
            "cohen_kappa": cohen_kappa(human, model), "confusion": confusion,
            "details": details}


def validation_markdown(v: dict[str, Any]) -> str:
    cols = (*LABELS, "unparseable")
    lines = [f"Judge agreed with human labels on {v['agreed']} of {v['n']} claims "
             f"({v['agreement']:.0%}); Cohen's kappa {v['cohen_kappa']:.2f}.", "",
             "| human \\ judge | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for h in LABELS:
        lines.append(f"| {h} | " + " | ".join(str(v["confusion"][h][m]) for m in cols) + " |")
    return "\n".join(lines)
