"""SQLite budget database built from extracted tables, plus a guarded read-only query runner.

Two layers:
  raw_cells    every cell of every table (long format), so no extracted figure is ever lost
  allocations  rows from tables recognised as budget allocations, normalised to TZS, with:
               - header synonyms in English and Swahili (Fungu, Taasisi, Kiasi, Jumla, ...)
               - units read from headers ("TZS milioni", "Shs '000") and applied
               - vote heading rows ("Vote 46 - Ministry of Education") carried down to projects
               - continuation tables (split across pages, header printed once) re-using the header
               - subtotal/total rows flagged, so SUM() never double counts
  vote_totals  view: totals per fiscal year and vote, excluding flagged total rows

The LLM never writes to this database: queries run on a read-only connection, must be a single
SELECT, and are stopped after a time limit.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Order matters: more specific fields are matched before general ones.
FIELD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("sub_vote", re.compile(r"sub[\s-]*vote|kifungu", re.I)),
    ("project_code", re.compile(r"(project|mradi).*(code|namba|no\b)|^code$|^namba", re.I)),
    ("project_name", re.compile(r"project|mradi", re.I)),
    ("vote_name", re.compile(r"vote\s*(name|desc)|institution|taasisi|ministry|wizara|jina",
                             re.I)),
    ("vote_code", re.compile(r"vote|fungu", re.I)),
    ("local", re.compile(r"local|ndani", re.I)),
    ("foreign", re.compile(r"foreign|nje|donor|wahisani", re.I)),
    ("total", re.compile(r"total|jumla|kiasi|amount|allocation|estimate", re.I)),
]
AMOUNT_FIELDS = ("local", "foreign", "total")
TOTAL_ROW = re.compile(r"^\s*(sub[\s-]*total|grand\s+total|total|jumla(\s+kuu)?)\b", re.I)
VOTE_CODE = re.compile(r"[\w./-]{1,8}")
VOTE_HEADING = re.compile(r"^\s*(vote|fungu)\s*[:.\-]?\s*(\w+)\s*[-:–—]?\s*(.*)$", re.I)
UNIT_PATTERNS = [
    (re.compile(r"bilion|billion|\bbn\b", re.I), 1e9),
    (re.compile(r"milion|million|\bmn\b|\bmill?\b", re.I), 1e6),
    # straight, curly and backtick apostrophes all appear in printed estimates: Shs '000 / ’000
    (re.compile(r"['’‘`´]\s*000|000s|thousand|elfu", re.I), 1e3),
]

SCHEMA = """
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY, title TEXT, doc_type TEXT, fiscal_year TEXT, language TEXT
);
CREATE TABLE raw_cells (
    table_id TEXT, doc_id TEXT, fiscal_year TEXT, page_no INTEGER,
    row_idx INTEGER, col_idx INTEGER, header TEXT, value TEXT, number REAL
);
CREATE TABLE allocations (
    id INTEGER PRIMARY KEY, doc_id TEXT, fiscal_year TEXT, page_no INTEGER, table_id TEXT,
    vote_code TEXT, vote_name TEXT, sub_vote TEXT, project_code TEXT, project_name TEXT,
    local_tzs REAL, foreign_tzs REAL, total_tzs REAL,
    unit_multiplier REAL, is_total_row INTEGER DEFAULT 0
);
CREATE TABLE projects (
    id INTEGER PRIMARY KEY, doc_id TEXT, fiscal_year TEXT, page_no INTEGER,
    vote_code TEXT, vote_name TEXT, sub_vote_code TEXT, sub_vote_name TEXT,
    project_code TEXT, project_name TEXT, actual_year TEXT, approved_year TEXT,
    actual_local_tzs REAL, actual_forex_tzs REAL, approved_local_tzs REAL,
    approved_forex_tzs REAL, est_local_tzs REAL, est_forex_tzs REAL, est_total_tzs REAL,
    funding TEXT, crd TEXT, donor TEXT
);
CREATE TABLE printed_totals (
    doc_id TEXT, page_no INTEGER, level TEXT, vote_code TEXT, sub_vote_code TEXT,
    est_local_tzs REAL, est_forex_tzs REAL, est_total_tzs REAL
);
CREATE TABLE vote_summary (
    doc_id TEXT, fiscal_year TEXT, page_no INTEGER, section TEXT, vote_code TEXT,
    vote_name TEXT, pe_tzs REAL, oc_tzs REAL, lga_oc_tzs REAL, dev_local_tzs REAL,
    dev_foreign_tzs REAL, lga_dev_tzs REAL, total_tzs REAL
);
CREATE INDEX ix_proj_vote ON projects (fiscal_year, vote_code);
CREATE INDEX ix_alloc_vote ON allocations (fiscal_year, vote_code);
CREATE VIEW project_votes AS
    SELECT doc_id, fiscal_year, vote_code, MAX(vote_name) AS vote_name,
           SUM(est_local_tzs) AS est_local_tzs, SUM(est_forex_tzs) AS est_forex_tzs,
           SUM(est_total_tzs) AS est_total_tzs, COUNT(*) AS n_projects,
           MIN(page_no) AS page_no, GROUP_CONCAT(DISTINCT page_no) AS pages
    FROM projects GROUP BY doc_id, fiscal_year, vote_code;
CREATE INDEX ix_cells_table ON raw_cells (table_id);
CREATE VIEW vote_totals AS
    SELECT doc_id, fiscal_year, vote_code, MAX(vote_name) AS vote_name,
           SUM(local_tzs) AS local_tzs, SUM(foreign_tzs) AS foreign_tzs,
           SUM(total_tzs) AS total_tzs, COUNT(*) AS n_rows,
           MIN(page_no) AS page_no, GROUP_CONCAT(DISTINCT page_no) AS pages
    FROM allocations WHERE is_total_row = 0
    GROUP BY doc_id, fiscal_year, vote_code;
"""

SCHEMA_NOTES = """\
Which table to use:
- vote_summary: one row per vote from the Volume IV 'approved budget by major categories' page:
  pe_tzs (personnel), oc_tzs (other charges), lga_oc_tzs, dev_local_tzs, dev_foreign_tzs,
  lga_dev_tzs and total_tzs. Use it for a vote's total, recurrent or development budget.
  Rows whose name starts 'Direct Transfer' carry a total only.
- projects: one row per development project in Volume IV, with its sub-vote, the estimate for
  the document's year (est_local_tzs, est_forex_tzs, est_total_tzs) and, for comparison, the
  approved estimate of the previous year (approved_year, approved_local_tzs,
  approved_forex_tzs) and actual spending two years earlier (actual_year, actual_*).
  funding is the printed Loan/Grant code; donor is the printed donor code.
- project_votes: projects summed per vote, with every page cited.
- For regional votes (RAS ...), the vote summary's development figure and the project pages'
  total are defined differently (transfers to local government authorities). Say which table
  a figure comes from.
- allocations / vote_totals: figures from tables inside the budget speeches.

Notes for writing queries:
- All money columns (*_tzs) are in Tanzanian shillings, already multiplied out from the units
  printed in the source tables.
- fiscal_year looks like '2025/26'. vote_code is the vote number as text, as printed:
  three digits in vote_summary and projects ('052' for Vote 52). To match any form, compare
  CAST(vote_code AS INTEGER) = 52.
- Rows with is_total_row = 1 are subtotals copied from the documents; exclude them when summing,
  or use the vote_totals view, which already does.
- The same vote can appear in several documents (a speech and the Volume IV estimates); filter
  or group by doc_id so figures from different documents are never added together.
- page_no and doc_id identify where each figure came from; include them so answers can cite.
  When you aggregate rows (SUM, COUNT), also select GROUP_CONCAT(DISTINCT page_no) AS pages so
  every page behind the total can be cited; the vote_totals view already has a pages column.
"""


def parse_number(value: str) -> float | None:
    v = value.strip().replace("−", "-")
    if not v or v in {"-", "–", "—", "nil", "N/A"}:
        return None
    negative = v.startswith("(") and v.endswith(")")
    v = v.strip("()").replace(",", "").replace(" ", "")
    if not re.fullmatch(r"-?\d+(\.\d+)?", v):
        return None
    n = float(v)
    return -n if negative else n


def detect_unit(headers: Iterable[str]) -> float:
    text = " ".join(headers)
    for pattern, multiplier in UNIT_PATTERNS:
        if pattern.search(text):
            return multiplier
    return 1.0


def _year_pattern(fiscal_year: str | None) -> re.Pattern[str] | None:
    """'2025/26' -> matches 2025/26, 2025-26, 2025/2026 in a header."""
    m = re.fullmatch(r"\s*(\d{4})\s*/\s*(\d{2,4})\s*", fiscal_year or "")
    if not m:
        return None
    return re.compile(rf"{m.group(1)}\s*[/-]\s*(?:\d{{2}})?{m.group(2)[-2:]}\b")


ANY_YEAR = re.compile(r"(?:19|20)\d{2}\s*[/-]\s*\d{2,4}")


def map_header(row: list[str], fiscal_year: str | None = None) -> dict[int, str]:
    """Map column index -> field name, each field used at most once.

    Estimates tables often have several amount columns (actual, approved, projected years).
    Among columns matching an amount field, prefer the one naming the document's own fiscal
    year, then one naming no year, then the first.
    """
    mapping: dict[int, str] = {}
    used: set[str] = set()
    texts = {i: c for i, c in enumerate(row) if c and parse_number(c) is None}
    for idx, cell in texts.items():                       # identifying fields first
        for name, pattern in FIELD_PATTERNS:
            if name in AMOUNT_FIELDS:
                continue
            if name not in used and pattern.search(cell):
                mapping[idx] = name
                used.add(name)
                break
    year = _year_pattern(fiscal_year)
    patterns = dict(FIELD_PATTERNS)
    for name in AMOUNT_FIELDS:                            # local, foreign, total in that order
        candidates = [i for i, c in texts.items()
                      if i not in mapping and patterns[name].search(c)]
        if not candidates:
            continue

        def rank(i: int) -> tuple[int, int]:
            cell = texts[i]
            if year and year.search(cell):
                return (0, i)
            return (1 if not ANY_YEAR.search(cell) else 2, i)

        mapping[min(candidates, key=rank)] = name
    return dict(sorted(mapping.items()))


def is_allocation_header(mapping: dict[int, str]) -> bool:
    fields = set(mapping.values())
    has_amount = bool(fields & set(AMOUNT_FIELDS))
    has_key = bool(fields & {"vote_code", "vote_name", "project_name", "project_code"})
    return has_amount and has_key


@dataclass
class _DocState:
    """Per-document context carried across tables (they often continue over pages)."""
    header: list[str] | None = None
    mapping: dict[int, str] = field(default_factory=dict)
    unit: float = 1.0
    vote: tuple[str, str] | None = None   # (code, name) from the latest vote heading row
    last_page: int = -10                  # page of the last allocation table seen


def _allocation_rows(table: dict[str, Any], state: _DocState) -> list[dict[str, Any]]:
    rows: list[list[str]] = table["rows"]
    mapping = map_header(rows[0], table.get("fiscal_year"))
    page = table["page_no"]
    if is_allocation_header(mapping):
        if rows[0] != state.header:     # a new kind of table; a repeated header keeps the vote
            state.vote = None
        state.header, state.mapping, state.unit = rows[0], mapping, detect_unit(rows[0])
        body = rows[1:]
    elif (state.header and len(rows[0]) == len(state.header)
          and page - state.last_page <= 1):
        body = rows                     # continuation: next page, same width, no header
    else:
        return []
    state.last_page = page

    out = []
    for row in body:
        cells = {name: (row[i] if i < len(row) else "") for i, name in state.mapping.items()}
        first = next((c for c in row if c), "")
        amounts = {f: parse_number(cells.get(f, "")) for f in AMOUNT_FIELDS}

        heading = VOTE_HEADING.match(first)
        if heading and all(v is None for v in amounts.values()):
            state.vote = (heading.group(2), heading.group(3).strip())
            continue
        if all(v is None for v in amounts.values()):
            continue                    # notes, blank or label-only rows

        local, foreign, total = amounts["local"], amounts["foreign"], amounts["total"]
        if total is None and (local is not None or foreign is not None):
            total = (local or 0) + (foreign or 0)
        # A real vote code is short ("46", "2006"); text like "Total Vote 46" in the Vote
        # column is a label, so fall back to the vote from the latest heading row.
        raw_code = cells.get("vote_code", "")
        code_ok = bool(VOTE_CODE.fullmatch(raw_code))
        vote_code = raw_code if code_ok else (state.vote[0] if state.vote else None)
        vote_name = cells.get("vote_name") or (state.vote[1] if state.vote else None)
        label = cells.get("project_name") or cells.get("vote_name") or first
        is_total = bool(TOTAL_ROW.match(label or "") or TOTAL_ROW.match(first))
        u = state.unit
        out.append({
            "doc_id": table["doc_id"], "fiscal_year": table["fiscal_year"],
            "page_no": table["page_no"], "table_id": table["table_id"],
            "vote_code": vote_code or None, "vote_name": vote_name or None,
            "sub_vote": cells.get("sub_vote") or None,
            "project_code": cells.get("project_code") or None,
            "project_name": cells.get("project_name") or None,
            "local_tzs": local * u if local is not None else None,
            "foreign_tzs": foreign * u if foreign is not None else None,
            "total_tzs": total * u if total is not None else None,
            "unit_multiplier": u,
            "is_total_row": int(is_total),
        })
    return out


def build_budget_db(
    tables: list[dict[str, Any]],
    db_path: str | Path,
    documents: Iterable[dict[str, Any]] = (),
    estimates: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, int]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)

    conn = sqlite3.connect(tmp)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO documents VALUES (:doc_id, :title, :doc_type, :fiscal_year, :language)",
        [{k: d.get(k) for k in ("doc_id", "title", "doc_type", "fiscal_year", "language")}
         for d in documents],
    )
    states: dict[str, _DocState] = {}
    n_cells = n_alloc = 0
    for t in sorted(tables, key=lambda t: (t["doc_id"], t["page_no"], t["table_id"])):
        header = t["rows"][0]
        cells = [
            (t["table_id"], t["doc_id"], t["fiscal_year"], t["page_no"], r, c,
             header[c] if c < len(header) else "", v, parse_number(v))
            for r, row in enumerate(t["rows"]) for c, v in enumerate(row)
        ]
        conn.executemany("INSERT INTO raw_cells VALUES (?,?,?,?,?,?,?,?,?)", cells)
        n_cells += len(cells)
        # Volume IV figures come from the layout-specific parser; generic table extraction on
        # those documents would only add partial duplicates.
        if t.get("doc_type") == "estimates" and estimates:
            continue
        alloc = _allocation_rows(t, states.setdefault(t["doc_id"], _DocState()))
        if alloc:
            cols = list(alloc[0])
            conn.executemany(
                f"INSERT INTO allocations ({', '.join(cols)}) "
                f"VALUES ({', '.join(':' + c for c in cols)})",
                alloc,
            )
            n_alloc += len(alloc)
    counts = {}
    for table, rows in (estimates or {}).items():
        if table not in ("projects", "printed_totals", "vote_summary") or not rows:
            counts[table] = 0
            continue
        cols = list(rows[0])
        conn.executemany(f"INSERT INTO {table} ({', '.join(cols)}) "
                         f"VALUES ({', '.join(':' + c for c in cols)})", rows)
        counts[table] = len(rows)
    conn.commit()
    conn.close()
    tmp.replace(path)                   # atomic swap: queries never see a half-built database
    return {"tables": len(tables), "cells": n_cells, "allocation_rows": n_alloc,
            "projects": counts.get("projects", 0), "vote_summary": counts.get("vote_summary", 0)}


# --- read-only querying (used by the query_budget tool) -------------------------------------


class UnsafeQueryError(ValueError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool

    def to_markdown(self, max_rows: int = 50) -> str:
        if not self.columns:
            return "(no columns)"
        lines = ["| " + " | ".join(self.columns) + " |", "|" + "---|" * len(self.columns)]
        for row in self.rows[:max_rows]:
            lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
        if self.truncated or len(self.rows) > max_rows:
            lines.append(f"(showing {min(len(self.rows), max_rows)} rows; more were available)")
        if not self.rows:
            lines.append("(no rows)")
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.0f}" if v.is_integer() else f"{v:,.2f}"
    return "" if v is None else str(v)


STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)


def _check_sql(sql: str) -> str:
    stripped = sql.strip().rstrip(";").strip()
    # Inspect the query with comments removed and string contents blanked, so a value such as
    # LIKE '%Create%' or 'a;b' is not mistaken for a keyword or a second statement.
    code = STRING_LITERAL.sub("''", COMMENT.sub(" ", stripped)).strip()
    if not re.match(r"^(select|with)\b", code, re.I):
        raise UnsafeQueryError("only SELECT queries are allowed")
    if ";" in code:
        raise UnsafeQueryError("only one statement is allowed")
    # Defence in depth: the connection is read-only anyway. REPLACE is allowed because
    # replace() is a legitimate string function inside a SELECT.
    if re.search(r"\b(attach|detach|pragma|insert|update|delete|drop|alter|create|vacuum)\b",
                 code, re.I):
        raise UnsafeQueryError("query contains a forbidden keyword")
    return stripped


def run_readonly(db_path: str | Path, sql: str, max_rows: int = 200,
                 timeout_s: float = 5.0) -> QueryResult:
    query = _check_sql(sql)
    if not Path(db_path).exists():
        raise FileNotFoundError(f"{db_path} not found; run `make index`")
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = conn.execute(query)
        rows = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in cur.description or []]
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc):
            raise TimeoutError(f"query exceeded {timeout_s}s") from exc
        raise
    finally:
        conn.close()
    return QueryResult(columns, rows[:max_rows], truncated=len(rows) > max_rows)


def schema_description(db_path: str | Path) -> str:
    """Schema plus live facts (years, row counts) for the LLM's tool description."""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        ddl = [r[0] for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL")]
        years = [r[0] for r in conn.execute(
            "SELECT DISTINCT fiscal_year FROM allocations ORDER BY 1")]
        n = conn.execute("SELECT COUNT(*) FROM allocations").fetchone()[0]
        n_proj = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        proj_years = [r[0] for r in conn.execute(
            "SELECT DISTINCT fiscal_year FROM projects ORDER BY 1")]
    finally:
        conn.close()
    facts = (f"allocations has {n} rows covering fiscal years: {', '.join(years) or 'none'}. "
             f"projects has {n_proj} rows covering: {', '.join(proj_years) or 'none'}.")
    return "\n\n".join(ddl) + "\n\n" + facts + "\n" + SCHEMA_NOTES


def load_tables(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
