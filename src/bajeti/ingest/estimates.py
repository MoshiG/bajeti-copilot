"""Parser for the Volume IV development estimates (Public Expenditure Estimates, Development Votes).

These tables are laid out with spaces, not ruling lines, so general table extraction misses
them. Their layout is regular, so it is parsed from word positions instead:

  Vote 036 RAS Katavi                                   page header: the current vote
  Sub Vote 8098 TRANSFERS TO LGAS - ...                 sub-vote heading
  6531 Project Monitoring and Evaluation                project line: code + name
  0  0  0  0  380,000,000  0  L  T  0GT  380,000,000    amounts line: actual (local, forex),
                                                        approved (local, forex), estimate
                                                        (local, forex), funding, C/R/D, donor,
                                                        total of the estimate
  Total of Subvote ...   Total of Vote ...              printed totals (7 amounts)

The vote summary near the front ("APPROVED BUDGET BY MAJOR CATEGORIES") is parsed too: one
row per vote with recurrent (PE, OC, LGA own-source OC) and development (local, foreign, LGA
own-source) estimates and the total.

Every figure keeps its page number, so the SQL tool can cite it. reconcile() checks parsed
projects against the printed sub-vote and vote totals; it is the parser's accuracy test.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AMOUNT = re.compile(r"^-?[\d,]+(\.\d+)?$|^-$|^\([\d,]+(\.\d+)?\)$")
VOTE_HEADER = re.compile(r"^Vote\s+(\w{2,4})\s+(.+)$")
SUB_VOTE = re.compile(r"^Sub\s+Vote\s+(\w+)\s+(.+)$", re.I)
PROJECT = re.compile(r"^(\d{4})\s+(\S.*)$")
YEAR = re.compile(r"(?:19|20)\d{2}/(?:19|20)?\d{2}")
SUMMARY_ROW = re.compile(r"^([0-9A-Z]{2,4})\s*-\s*(.+)$")
LINE_TOLERANCE = 3.0          # points; words whose baselines differ by less share a line


def _amount(token: str) -> float | None:
    """'1,250,000' -> 1250000.0; '-' -> 0.0; '(300,000)' -> -300000.0; text -> None."""
    if token == "-":
        return 0.0
    if not AMOUNT.match(token):
        return None
    negative = token.startswith("(")
    value = float(token.strip("()").replace(",", ""))
    return -value if negative else value


@dataclass
class Line:
    words: list[str]
    xs: list[float]                      # left edge of each word
    ends: list[float] = field(default_factory=list)   # right edge of each word

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def amounts(self) -> list[float]:
        return [a for a in (_amount(w) for w in self.words) if a is not None]


def page_lines(page: Any) -> list[Line]:
    """Rebuild text lines from word boxes (PyMuPDF), left to right."""
    rows: list[list[tuple]] = []
    for w in sorted(page.get_text("words"), key=lambda w: (round(w[3], 1), w[0])):
        if rows and abs(rows[-1][0][3] - w[3]) <= LINE_TOLERANCE:
            rows[-1].append(w)
        else:
            rows.append([w])
    out: list[Line] = []
    for r in rows:
        r.sort(key=lambda w: w[0])
        line = Line([w[4] for w in r], [w[0] for w in r], [w[2] for w in r])
        # A total too wide for its column wraps onto the next line as a lone amount; rejoin it.
        # (Page numbers are lone numbers too, but small, uncommaed and centred.)
        if (out and len(line.words) == 1 and "," in line.words[0]
                and _amount(line.words[0]) is not None and len(out[-1].amounts()) == 6):
            out[-1].words.append(line.words[0])
            out[-1].xs.append(line.xs[0])
            out[-1].ends.append(line.ends[0])
            continue
        out.append(line)
    return out


@dataclass
class Project:
    doc_id: str
    fiscal_year: str
    page_no: int
    vote_code: str
    vote_name: str
    sub_vote_code: str | None
    sub_vote_name: str | None
    project_code: str
    project_name: str
    actual_year: str | None
    approved_year: str | None
    actual_local_tzs: float
    actual_forex_tzs: float
    approved_local_tzs: float
    approved_forex_tzs: float
    est_local_tzs: float
    est_forex_tzs: float
    est_total_tzs: float
    funding: str | None          # L (loan) / G (grant) etc., as printed
    crd: str | None              # the C/R/D column, as printed
    donor: str | None


@dataclass
class PrintedTotal:
    doc_id: str
    page_no: int
    level: str                   # subvote | vote
    vote_code: str
    sub_vote_code: str | None
    est_local_tzs: float
    est_forex_tzs: float
    est_total_tzs: float


@dataclass
class VoteSummary:
    doc_id: str
    fiscal_year: str
    page_no: int
    section: str | None
    vote_code: str
    vote_name: str
    pe_tzs: float
    oc_tzs: float
    lga_oc_tzs: float
    dev_local_tzs: float
    dev_foreign_tzs: float
    lga_dev_tzs: float
    total_tzs: float


@dataclass
class EstimatesResult:
    projects: list[Project] = field(default_factory=list)
    totals: list[PrintedTotal] = field(default_factory=list)
    summary: list[VoteSummary] = field(default_factory=list)
    unparsed: list[tuple[int, str]] = field(default_factory=list)   # (page, line) for review

    def to_records(self) -> dict[str, list[dict[str, Any]]]:
        return {"projects": [asdict(p) for p in self.projects],
                "printed_totals": [asdict(t) for t in self.totals],
                "vote_summary": [asdict(s) for s in self.summary]}


SUMMARY_FIELDS = ("pe_tzs", "oc_tzs", "lga_oc_tzs", "dev_local_tzs", "dev_foreign_tzs",
                  "lga_dev_tzs", "total_tzs")
NAME_GAP = 20.0        # points; a wider gap than this ends the vote name and starts the amounts
COLUMN_MERGE = 15.0    # right edges closer than this belong to the same column
HEADER_WORDS = re.compile(r"emoluments|\((PE|OC)\)|ownsource|expenditure|charges|code and name|"
                          r"^(grand\s+)?(sub[\s-]*)?total|recurrent|development\s+development",
                          re.I)


def _split_name(ln: Line) -> tuple[str, str, int] | None:
    """'003 - National Land Use Planning Commission   4,423,468,000 ...' -> (code, name, index
    of the first value cell). Names can contain ' - ', so the boundary is the first wide gap.
    The dash is sometimes printed against the code ('052- Direct Transfer to ...')."""
    if not SUMMARY_ROW.match(ln.text):
        return None
    if len(ln.words) >= 3 and ln.words[1] == "-":
        code, start = ln.words[0], 2
    elif len(ln.words) >= 2 and re.fullmatch(r"[0-9A-Z]{2,4}-", ln.words[0]):
        code, start = ln.words[0][:-1], 1
    else:
        return None
    split = len(ln.words)
    for i in range(start + 1, len(ln.words)):
        if ln.xs[i] - ln.ends[i - 1] > NAME_GAP:
            split = i
            break
    return code, " ".join(ln.words[start:split]), split


def _is_amounts_only(ln: Line) -> bool:
    return bool(ln.words) and all(_amount(w) is not None for w in ln.words)


def _column_edges(rows: list[tuple[list[str], list[float]]]) -> list[float] | None:
    """Right edges of the 7 summary columns on a page, found by grouping where numbers end.
    Amounts are right-aligned, so every number in a column ends at (almost) the same x."""
    edges = sorted(e for words, ends in rows for w, e in zip(words, ends, strict=True)
                   if _amount(w) is not None and any(c.isdigit() for c in w))
    clusters: list[list[float]] = []
    for e in edges:
        if clusters and e - clusters[-1][-1] <= COLUMN_MERGE:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    if len(clusters) != len(SUMMARY_FIELDS):
        return None
    return [sum(c) / len(c) for c in clusters]


SUMMARY_CODES = ("101", "102", "103", "201", "202", "203", "Total")


def _header_anchors(lines: list[Line]) -> list[float] | None:
    """Left edges of the printed column codes 101, 102, 103, 201, 202, 203 and Total."""
    for ln in lines[:10]:
        pos = {w: x for w, x in zip(ln.words, ln.xs, strict=True) if w in SUMMARY_CODES}
        if len(pos) == len(SUMMARY_CODES):
            return [pos[c] for c in SUMMARY_CODES]
    return None


def _place_by_anchors(words: list[str], ends: list[float], anchors: list[float]) -> list[float]:
    """A right-aligned amount ends just before the next column's code: column i is the first
    whose following anchor lies at or beyond the amount's right edge."""
    values = [0.0] * len(anchors)
    slack = 2.0
    for w, e in zip(words, ends, strict=True):
        amount = _amount(w)
        if amount is None:
            continue
        col = next((i for i in range(len(anchors) - 1) if e <= anchors[i + 1] + slack),
                   len(anchors) - 1)
        values[col] += amount
    return values


def _place(words: list[str], ends: list[float], edges: list[float]) -> list[float]:
    values = [0.0] * len(edges)
    for w, e in zip(words, ends, strict=True):
        amount = _amount(w)
        if amount is not None:
            col = min(range(len(edges)), key=lambda i: abs(edges[i] - e))
            values[col] += amount
    return values


def _parse_summary_page(lines: list[Line], doc_id: str, fy: str, page_no: int,
                        out: EstimatesResult) -> None:
    """One row per vote: code, name, then PE, OC, LGA own-source OC, development local and
    foreign, LGA own-source development, total. Empty cells are '-' in some years and blank in
    others, and long names wrap, so rows are rebuilt first and amounts placed by position:
    by the printed column codes when present, else by grouping where numbers end, else (when
    every cell is printed) in order."""
    rows: list[dict[str, Any]] = []
    section = None
    attached = False          # the previous line belonged to the last row
    for ln in lines:
        if re.match(r"^[A-Z]:\s", ln.text):
            section, attached = ln.text, False
            continue
        if parts := _split_name(ln):
            code, name, split = parts
            rows.append({"code": code, "name": name, "section": section,
                         "words": ln.words[split:], "ends": ln.ends[split:]})
            attached = True
        elif attached and _is_amounts_only(ln) and not rows[-1]["words"]:
            rows[-1]["words"], rows[-1]["ends"] = ln.words, ln.ends    # amounts on next line
        elif (attached and len(ln.words) <= 6
              and not any(_amount(w) is not None for w in ln.words)
              and not HEADER_WORDS.search(ln.text)):
            rows[-1]["name"] += " " + ln.text                         # name continued
        else:
            attached = False
    anchors = _header_anchors(lines)
    edges = None if anchors else _column_edges([(r["words"], r["ends"]) for r in rows])
    for r in rows:
        if not r["words"]:
            out.unparsed.append((page_no, f"{r['code']} - {r['name']}"))
            continue
        if anchors:
            values = _place_by_anchors(r["words"], r["ends"], anchors)
        elif edges:
            values = _place(r["words"], r["ends"], edges)
        else:
            amounts = [a for a in (_amount(w) for w in r["words"]) if a is not None]
            if len(amounts) != len(SUMMARY_FIELDS):
                out.unparsed.append((page_no, f"{r['code']} - {r['name']}"))
                continue
            values = amounts
        out.summary.append(VoteSummary(doc_id, fy, page_no, r["section"], r["code"],
                                       r["name"].strip(), *values))


def parse_estimates(path: str | Path, doc_id: str, fiscal_year: str) -> EstimatesResult:
    import pymupdf

    out = EstimatesResult()
    vote_code = vote_name = None
    sub_code = sub_name = None
    actual_year = approved_year = None
    pending: tuple[str, str, int] | None = None          # (code, name, page) awaiting amounts
    last_project: tuple[str, str] | None = None          # for amounts continued on a new page

    with pymupdf.open(path) as doc:
        for idx, page in enumerate(doc):
            page_no = idx + 1
            lines = page_lines(page)
            if not lines:
                continue
            head = " ".join(ln.text for ln in lines[:4])
            if "BY MAJOR CATEGORIES" in head.upper():
                _parse_summary_page(lines, doc_id, fiscal_year, page_no, out)
                continue

            # A vote's first project page opens with an introduction ("A. ESTIMATE of the
            # amount required ...") before the table, so the header can sit anywhere on it.
            if not any(ln.words[:2] == ["Item", "Description"] for ln in lines):
                continue
            in_table = False
            for ln in lines:
                text = ln.text
                if m := VOTE_HEADER.match(text):
                    if m.group(1) != vote_code:
                        sub_code = sub_name = None
                        pending = last_project = None
                    vote_code, vote_name = m.group(1), m.group(2).strip()
                    continue
                years = YEAR.findall(text)
                if len(years) >= 2 and not ln.amounts():
                    actual_year, approved_year = years[0], years[1]
                if ln.words[:2] == ["Item", "Description"]:
                    in_table = True
                    continue
                if not in_table:
                    continue
                if text.startswith(("Actual", "Local", "Shs")):
                    continue
                if m := SUB_VOTE.match(text):
                    sub_code, sub_name = m.group(1), m.group(2).strip()
                    pending = last_project = None
                    continue
                if text.startswith("Total of"):
                    values = ln.amounts()
                    if len(values) >= 7 and vote_code:
                        level = "vote" if text.startswith("Total of Vote") else "subvote"
                        out.totals.append(PrintedTotal(
                            doc_id, page_no, level, vote_code,
                            sub_code if level == "subvote" else None,
                            values[4], values[5], values[-1]))
                    pending = None
                    continue

                values = ln.amounts()
                codes = [w for w in ln.words if _amount(w) is None]
                if len(values) >= 7 and len(codes) <= 3:
                    target = pending or (last_project + (page_no,) if last_project else None)
                    if target and vote_code:
                        code, name, p_page = target
                        funding, crd, donor = (codes + [None] * 3)[:3]
                        out.projects.append(Project(
                            doc_id, fiscal_year, page_no, vote_code, vote_name, sub_code,
                            sub_name, code, name, actual_year, approved_year,
                            *values[:6], values[-1], funding, crd, donor))
                        last_project = (code, name)
                    else:
                        out.unparsed.append((page_no, text))
                    continue
                if m := PROJECT.match(text):
                    pending = (m.group(1), m.group(2).strip(), page_no)
                    last_project = None
                    continue
                if pending and not values:          # project name wrapped onto a second line
                    code, name, p_page = pending
                    pending = (code, f"{name} {text}", p_page)
                    continue
                if text.strip() and not text.strip().isdigit():
                    out.unparsed.append((page_no, text))
    return out


def reconcile(result: EstimatesResult, tolerance: float = 1.0) -> dict[str, Any]:
    """Compare parsed project sums with every printed sub-vote and vote total."""
    from collections import defaultdict

    by_sub: dict[tuple, float] = defaultdict(float)
    by_vote: dict[str, float] = defaultdict(float)
    for p in result.projects:
        by_sub[(p.vote_code, p.sub_vote_code)] += p.est_total_tzs
        by_vote[p.vote_code] += p.est_total_tzs
    checks = {"subvote": [0, 0], "vote": [0, 0]}
    mismatches = []
    for t in result.totals:
        parsed = by_sub[(t.vote_code, t.sub_vote_code)] if t.level == "subvote" \
            else by_vote[t.vote_code]
        ok = abs(parsed - t.est_total_tzs) <= tolerance
        checks[t.level][0] += ok
        checks[t.level][1] += 1
        if not ok:
            mismatches.append({"level": t.level, "vote": t.vote_code, "sub_vote": t.sub_vote_code,
                               "page": t.page_no, "printed": t.est_total_tzs, "parsed": parsed})
    # Each summary row must add up to its own printed total. (Summary development figures are
    # not compared with project pages: for regional votes the two tables define them
    # differently, e.g. transfers to local government authorities.)
    def adds_up(r: VoteSummary) -> bool:
        parts = [getattr(r, f) for f in SUMMARY_FIELDS[:-1]]
        # Direct transfers are printed with a total only ('-' in every other column).
        return abs(sum(parts) - r.total_tzs) <= tolerance or not any(parts)

    summary_ok = sum(adds_up(r) for r in result.summary)
    return {"projects": len(result.projects), "votes": len(by_vote),
            "summary_rows": len(result.summary), "unparsed_lines": len(result.unparsed),
            "subvote_totals_matched": checks["subvote"], "vote_totals_matched": checks["vote"],
            "summary_rows_add_up": [summary_ok, len(result.summary)],
            "mismatches": mismatches}
