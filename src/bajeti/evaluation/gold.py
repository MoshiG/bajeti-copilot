"""Gold questions: the format, loading, and a validator that catches mistakes before GPU time.

One JSON object per line in eval/gold.jsonl:

{
  "id": "q012",
  "type": "numeric",          # factual | numeric | comparison | cross_lingual | unanswerable
  "language": "sw",                     # language of the question
  "question": "Wizara ya Afya imetengewa kiasi gani cha fedha za maendeleo 2025/26?",
  "answerable": true,
  "answer": "TZS ... (reference answer, for humans reading the results)",
  "numbers": [980500000000],            # figures a correct answer must state (TZS, any unit ok)
  "keywords": ["Wizara ya Afya"],       # phrases a correct answer must contain
  "evidence": [{"doc_id": "volume_iv_2025_26", "pages": [212, 213]}],
  "split": "test",                     # dev (tune prompts on these) | test (report these)
  "notes": "optional"
}

Scoring is deterministic (numbers within 0.5%, keywords case-insensitive), so results are
reproducible and cost nothing to re-score. Any field containing "TODO" marks the item as a
draft: drafts are reported by the validator and skipped by the evaluation runner.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TYPES = ("factual", "numeric", "comparison", "cross_lingual", "unanswerable")


@dataclass
class Evidence:
    doc_id: str
    pages: list[int]


@dataclass
class GoldItem:
    id: str
    type: str
    language: str
    question: str
    answerable: bool = True
    answer: str = ""
    numbers: list[float] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    split: str = "test"
    notes: str = ""

    @property
    def is_draft(self) -> bool:
        return "TODO" in json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldItem:
        d = dict(d)
        d["evidence"] = [Evidence(e["doc_id"], [int(p) for p in e.get("pages", [])])
                         for e in d.get("evidence", [])]
        d["numbers"] = [float(n) for n in d.get("numbers", []) if not isinstance(n, str)]
        return cls(**d)


def load_gold(path: str | Path) -> list[GoldItem]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if line.strip() and not line.lstrip().startswith("//"):
                try:
                    items.append(GoldItem.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    raise ValueError(f"{path}:{n}: {exc}") from exc
    return items


@dataclass
class GoldReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, Counter] = field(default_factory=dict)
    ready: int = 0
    drafts: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_gold(items: list[GoldItem], doc_ids: set[str] | None = None) -> GoldReport:
    r = GoldReport(counts={"type": Counter(), "language": Counter(), "split": Counter()})
    seen: set[str] = set()
    for it in items:
        where = f"{it.id}:"
        if it.id in seen:
            r.errors.append(f"{where} duplicate id")
        seen.add(it.id)
        if it.type not in TYPES:
            r.errors.append(f"{where} type must be one of {TYPES}")
        if it.language not in {"sw", "en"}:
            r.errors.append(f"{where} language must be 'sw' or 'en'")
        if it.split not in {"dev", "test"}:
            r.errors.append(f"{where} split must be 'dev' or 'test'")
        if not it.question.strip():
            r.errors.append(f"{where} empty question")
        if (it.type == "unanswerable") == it.answerable:
            r.errors.append(f"{where} type 'unanswerable' must go with answerable=false")
        if it.is_draft:
            r.drafts += 1
            continue
        if it.answerable:
            if not it.evidence:
                r.errors.append(f"{where} answerable question needs evidence pages")
            if not (it.numbers or it.keywords):
                r.errors.append(f"{where} needs numbers or keywords to score the answer")
            if it.type in {"numeric", "comparison"} and not it.numbers:
                r.warnings.append(f"{where} {it.type} question without expected numbers")
        elif it.evidence or it.numbers or it.keywords:
            r.warnings.append(f"{where} unanswerable question has answer fields (ignored)")
        for ev in it.evidence:
            if doc_ids is not None and ev.doc_id not in doc_ids:
                r.errors.append(f"{where} unknown doc_id {ev.doc_id!r}")
            if not ev.pages:
                r.errors.append(f"{where} evidence for {ev.doc_id} lists no pages")
        r.counts["type"][it.type] += 1
        r.counts["language"][it.language] += 1
        r.counts["split"][it.split] += 1
        r.ready += 1
    if r.ready and r.counts["type"]["unanswerable"] / r.ready < 0.1:
        r.warnings.append("fewer than 10% unanswerable questions: refusal behaviour is "
                          "barely measured")
    return r
