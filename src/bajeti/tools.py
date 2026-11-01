"""The copilot's three tools, and the source registry that makes citations checkable.

search_docs         hybrid retrieval over the documents, optional year/type/language
query_budget        read-only SQL over the budget database for exact figures
rank_interventions  exact budget-ceiling optimiser for prioritisation questions

Every piece of evidence a tool returns is registered under an id (S1, S2, ...). The model must
cite those ids; the agent then checks each citation against the registry, so a citation in the
final answer always points to something a tool really returned.

Tools never raise into the agent loop: errors come back as text the model can react to (fix
the SQL, drop a filter), which is how small open models recover in practice.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bajeti.config import Config
from bajeti.index.budget_db import UnsafeQueryError, run_readonly, schema_description
from bajeti.optimize import Intervention, rank_interventions
from bajeti.retrieval import Filters, Retriever


@dataclass
class Source:
    id: str                    # S1, S2, ...
    citation: str              # "Hotuba ya Bajeti 2025/26, p. 2"
    kind: str                  # document | table_row
    doc_id: str
    page: int | None = None
    chunk_id: str | None = None
    excerpt: str = ""


class SourceRegistry:
    """Assigns stable ids to evidence across all tool calls in one answer."""

    def __init__(self) -> None:
        self.sources: dict[str, Source] = {}
        self._keys: dict[tuple, str] = {}

    def add(self, key: tuple, **kw: Any) -> Source:
        if key in self._keys:                       # same chunk twice -> same id
            return self.sources[self._keys[key]]
        sid = f"S{len(self.sources) + 1}"
        src = Source(id=sid, **kw)
        self.sources[sid], self._keys[key] = src, sid
        return src

    def __contains__(self, sid: str) -> bool:
        return sid in self.sources


@dataclass
class ToolResult:
    content: str
    error: bool = False
    source_ids: list[str] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any], SourceRegistry], ToolResult]

    def spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


# --- search_docs -------------------------------------------------------------------------------


def make_search_docs(retriever: Retriever, years: list[str]) -> Tool:
    def run(args: dict[str, Any], reg: SourceRegistry) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult("Error: 'query' is required.", error=True)
        filters = Filters(args.get("fiscal_year"), args.get("doc_type"), args.get("language"))
        hits = retriever.search(query, filters=filters)
        if not hits and filters.active():
            return ToolResult(f"No passages matched with filters {filters}. Try again without "
                              "filters or with a different fiscal year.", error=False)
        if not hits:
            return ToolResult("No passages matched. Rephrase the query or try Swahili/English.")
        blocks, ids = [], []
        for h in hits:
            c = h.chunk
            src = reg.add(("chunk", c.chunk_id), citation=c.citation, kind="document",
                          doc_id=c.doc_id, page=c.page_start, chunk_id=c.chunk_id,
                          excerpt=c.text.split("\n", 1)[-1][:2000])
            body = c.text.split("\n", 1)[-1]
            where = f" — {c.section}" if c.section else ""
            blocks.append(f"[{src.id}] {c.citation}{where} ({c.fiscal_year}, {c.language})\n{body}")
            ids.append(src.id)
        return ToolResult("\n\n".join(blocks), source_ids=ids)

    return Tool(
        name="search_docs",
        description=(
            "Search Tanzania's budget speeches, Finance Acts and expenditure estimates for "
            "passages on policy, measures, tax changes and narrative explanations. Returns "
            "numbered passages [S#] to cite. Works with Swahili or English queries. "
            f"Fiscal years available: {', '.join(years) or 'unknown'}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "fiscal_year": {"type": "string", "description": "Optional, e.g. '2025/26'."},
                "doc_type": {"type": "string", "enum": ["speech", "finance_act", "estimates"]},
                "language": {"type": "string", "enum": ["sw", "en"]},
            },
            "required": ["query"],
        },
        run=run,
    )


# --- query_budget ------------------------------------------------------------------------------

MAX_SQL_SOURCES = 15


def _doc_titles(db_path: Path) -> dict[str, str]:
    try:
        rows = run_readonly(db_path, "SELECT doc_id, title FROM documents").rows
    except Exception:
        return {}
    return {d: t for d, t in rows if t}


def make_query_budget(db_path: str | Path) -> Tool:
    db_path = Path(db_path)
    titles = _doc_titles(db_path)

    def run(args: dict[str, Any], reg: SourceRegistry) -> ToolResult:
        sql = str(args.get("sql", "")).strip()
        if not sql:
            return ToolResult("Error: 'sql' is required.", error=True)
        try:
            result = run_readonly(db_path, sql, max_rows=100)
        except UnsafeQueryError as exc:
            return ToolResult(f"Query rejected: {exc}. Write a single SELECT.", error=True)
        except TimeoutError:
            return ToolResult("Query took too long. Add filters (fiscal_year, vote_code) "
                              "or aggregate with GROUP BY.", error=True)
        except Exception as exc:  # sqlite3 syntax errors, unknown columns, ...
            return ToolResult(f"SQL error: {exc}. Check table and column names in the schema.",
                              error=True)

        cols = [c.lower() for c in result.columns]
        ids: list[str] = []
        rows = [list(r) for r in result.rows]
        if "doc_id" in cols and ("page_no" in cols or "pages" in cols):
            di = cols.index("doc_id")
            for row in rows:
                if len(ids) >= MAX_SQL_SOURCES:
                    break
                doc = row[di]
                # An aggregate cites every page it sums (pages = "12,13,14"), not just one.
                if "pages" in cols and row[cols.index("pages")]:
                    pages = [int(p) for p in str(row[cols.index("pages")]).split(",")
                             if p.strip().isdigit()]
                else:
                    pages = [row[cols.index("page_no")]] if "page_no" in cols else []
                title = titles.get(doc, doc)
                row_text = "; ".join(f"{col}: {val}" for col, val in
                                     zip(result.columns, row, strict=False))
                row_ids = []
                for page in pages[: MAX_SQL_SOURCES - len(ids)] or []:
                    src = reg.add(("row", doc, page), citation=f"{title}, p. {page}",
                                  kind="table_row", doc_id=doc, page=page, excerpt=row_text)
                    if row_text not in src.excerpt:     # several rows from the same page
                        src.excerpt += "\n" + row_text
                    row_ids.append(src.id)
                    if src.id not in ids:
                        ids.append(src.id)
                row.append(" ".join(row_ids))
            result.columns.append("source")
            result.rows = [tuple(r + [""] * (len(result.columns) - len(r))) for r in rows]
            note = "Cite figures with the [S#] in the source column."
        else:
            note = ("These rows have no doc_id and page_no (or pages), so they cannot be cited. "
                    "Re-run the query selecting them before stating any figure.")
        legend = "\n".join(f"[{s}] {reg.sources[s].citation}" for s in ids)
        body = result.to_markdown()
        return ToolResult(f"{body}\n\n{note}" + (f"\n{legend}" if legend else ""),
                          source_ids=ids)

    try:
        schema = schema_description(db_path)
    except Exception:
        schema = "(budget database not built yet)"
    return Tool(
        name="query_budget",
        description=(
            "Run one read-only SQLite SELECT over budget allocation tables for exact figures, "
            "totals and comparisons. Use this instead of doing arithmetic yourself. Always "
            "select doc_id and page_no so figures can be cited.\n\nSchema:\n" + schema
        ),
        parameters={
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT query."}},
            "required": ["sql"],
        },
        run=run,
    )


# --- rank_interventions ------------------------------------------------------------------------


def make_rank_interventions() -> Tool:
    def run(args: dict[str, Any], reg: SourceRegistry) -> ToolResult:
        try:
            raw = args.get("interventions")
            if isinstance(raw, str):            # some models send nested JSON as a string
                raw = json.loads(raw)
            items = [Intervention(name=str(i["name"]), cost=float(i["cost"]),
                                  benefit=float(i["benefit"]),
                                  mandatory=bool(i.get("mandatory", False))) for i in raw or []]
            if not items:
                return ToolResult("Error: provide at least one intervention.", error=True)
            plan = rank_interventions(items, float(args["budget_ceiling"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return ToolResult(f"Error: {exc}. Each intervention needs name, cost, benefit; "
                              "budget_ceiling is required.", error=True)
        return ToolResult(plan.to_markdown() + "\n\nThis is a computation over the inputs you "
                          "supplied. Cite the sources of the costs, not this tool.")

    return Tool(
        name="rank_interventions",
        description=(
            "Choose which interventions to fund under a budget ceiling, maximising total "
            "benefit (exact optimisation). Use for prioritisation questions after you have "
            "found the costs with the other tools. Benefit is any comparable score the user "
            "provides or the documents support (beneficiaries, priority score). Costs and the "
            "ceiling must be in the same unit."
        ),
        parameters={
            "type": "object",
            "properties": {
                "interventions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "cost": {"type": "number"},
                            "benefit": {"type": "number"},
                            "mandatory": {"type": "boolean"},
                        },
                        "required": ["name", "cost", "benefit"],
                    },
                },
                "budget_ceiling": {"type": "number"},
            },
            "required": ["interventions", "budget_ceiling"],
        },
        run=run,
    )


def build_tools(cfg: Config, retriever: Retriever) -> list[Tool]:
    years = sorted({c.fiscal_year for c in retriever.indexes.store.chunks})
    tools = [make_search_docs(retriever, years)]
    if cfg.tools.enable_sql:
        tools.append(make_query_budget(cfg.paths.sqlite_path))
    if cfg.tools.enable_optimizer:
        tools.append(make_rank_interventions())
    return tools
