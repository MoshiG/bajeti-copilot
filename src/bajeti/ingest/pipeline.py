"""Run ingestion over the corpus and write everything indexing needs.

data/processed/
    pages/<doc_id>.jsonl         parsed pages (cached; re-parse with --force)
    chunks_naive.jsonl           input to ablation rows 1-3
    chunks_structured.jsonl      input to ablation rows 4-5
    tables.jsonl                 every extracted table, for the SQLite budget database
    ingest_report.json           per-document stats: pages, OCR'd pages, tables, chunks
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bajeti.config import Config
from bajeti.corpus import Document, Manifest
from bajeti.ingest.chunk import Chunk, chunk_pages
from bajeti.ingest.parse import Page, parse_pdf

log = logging.getLogger(__name__)
STRATEGIES = ("naive", "structured")


@dataclass
class DocReport:
    doc_id: str
    status: str = "ok"                  # ok | missing | failed
    pages: int = 0
    ocr_pages: int = 0
    tables: int = 0
    chunks: dict[str, int] = field(default_factory=dict)
    message: str = ""


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_or_parse(doc: Document, raw_dir: Path, pages_dir: Path, cfg: Config,
                  force: bool) -> list[Page]:
    cache = pages_dir / f"{doc.id}.jsonl"
    if cache.exists() and not force:
        return [Page.from_dict(d) for d in read_jsonl(cache)]
    pages = parse_pdf(raw_dir / doc.filename, doc.id, doc.language, cfg.ingest.ocr_fallback)
    _write_jsonl(cache, (p.to_dict() for p in pages))
    return pages


def ingest_corpus(
    cfg: Config,
    manifest: Manifest,
    only: set[str] | None = None,
    force: bool = False,
) -> list[DocReport]:
    raw_dir, out_dir = Path(cfg.paths.raw_dir), Path(cfg.paths.processed_dir)
    pages_dir = out_dir / "pages"
    all_chunks: dict[str, list[Chunk]] = {s: [] for s in STRATEGIES}
    tables: list[dict[str, Any]] = []
    reports: list[DocReport] = []

    for doc in manifest.documents:
        if only and doc.id not in only:
            continue
        report = DocReport(doc.id)
        if not (raw_dir / doc.filename).exists():
            report.status, report.message = "missing", "not downloaded; run `make fetch`"
            reports.append(report)
            continue
        try:
            pages = load_or_parse(doc, raw_dir, pages_dir, cfg, force)
        except Exception as exc:
            log.exception("parsing %s failed", doc.id)
            report.status, report.message = "failed", str(exc)
            reports.append(report)
            continue

        report.pages = len(pages)
        report.ocr_pages = sum(p.ocr for p in pages)
        for p in pages:
            for i, b in enumerate(b for b in p.blocks if b.kind == "table"):
                tables.append({
                    "table_id": f"{doc.id}:p{p.page_no}:t{i}",
                    "doc_id": doc.id, "fiscal_year": doc.fiscal_year, "doc_type": doc.doc_type,
                    "page_no": p.page_no, "rows": b.rows,
                })
                report.tables += 1
        for strategy in STRATEGIES:
            chunks = chunk_pages(pages, doc, strategy, cfg.ingest.chunk_tokens,
                                 cfg.ingest.chunk_overlap)
            all_chunks[strategy].extend(chunks)
            report.chunks[strategy] = len(chunks)
        reports.append(report)

    # Partial runs (--only) must not wipe other documents' outputs.
    if not only:
        for strategy, chunks in all_chunks.items():
            _write_jsonl(out_dir / f"chunks_{strategy}.jsonl", (c.to_dict() for c in chunks))
        _write_jsonl(out_dir / "tables.jsonl", tables)
    (out_dir / "ingest_report.json").write_text(
        json.dumps([r.__dict__ for r in reports], indent=2), encoding="utf-8"
    )
    return reports
