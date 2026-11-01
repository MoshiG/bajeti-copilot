"""Two chunking strategies, compared head-to-head in the ablation table.

naive       Fixed windows of N words with overlap, ignoring structure. Tables are flattened
            into the stream. This is what most RAG demos ship.
structured  Chunks follow the document: they break at headings, keep paragraphs whole, emit
            each table as its own chunk (header row repeated if split), and carry a section
            path such as "SEHEMU YA PILI > Mapato ya Kodi" for citations and embeddings.

Sizes are counted in whitespace-separated words, a stable proxy for model tokens that does not
tie the chunker to one tokenizer.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from bajeti.corpus import Document
from bajeti.ingest.parse import Page, table_to_markdown

MIN_SECTION_WORDS = 40   # sections shorter than this merge forward instead of standing alone


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    fiscal_year: str
    language: str
    strategy: str
    kind: str              # text | table
    section: str
    page_start: int
    page_end: int
    text: str              # what is embedded and shown to the LLM (includes a context header)
    n_words: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def citation(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"{self.title}, {pages}"


def _words(text: str) -> int:
    return len(text.split())


def _windows(words: list[Any], size: int, overlap: int) -> list[list[Any]]:
    if not words:
        return []
    step = max(size - overlap, 1)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


class _Builder:
    def __init__(self, doc: Document, strategy: str) -> None:
        self.doc, self.strategy, self.chunks = doc, strategy, []

    def emit(self, body: str, kind: str, section: str, p0: int, p1: int) -> None:
        body = body.strip()
        if not body:
            return
        header = f"[{self.doc.title}" + (f" | {section}" if section else "") + "]"
        self.chunks.append(Chunk(
            chunk_id=f"{self.doc.id}:{self.strategy[0]}{len(self.chunks):04d}",
            doc_id=self.doc.id,
            title=self.doc.title,
            doc_type=self.doc.doc_type,
            fiscal_year=self.doc.fiscal_year,
            language=self.doc.language,
            strategy=self.strategy,
            kind=kind,
            section=section,
            page_start=p0,
            page_end=p1,
            text=f"{header}\n{body}",
            n_words=_words(body),
        ))


def naive_chunks(pages: list[Page], doc: Document, size: int, overlap: int) -> list[Chunk]:
    tagged = [(w, p.page_no) for p in pages for b in p.blocks for w in b.text.split()]
    builder = _Builder(doc, "naive")
    for window in _windows(tagged, size, overlap):
        builder.emit(" ".join(w for w, _ in window), "text", "", window[0][1], window[-1][1])
    return builder.chunks


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Split an over-long paragraph at sentence ends, falling back to word windows."""
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    parts, current = [], []
    for s in sentences:
        if current and _words(" ".join(current + [s])) > size:
            parts.append(" ".join(current))
            tail = " ".join(current).split()[-overlap:] if overlap else []
            current = [" ".join(tail)] if tail else []
        current.append(s)
    if current:
        parts.append(" ".join(current))
    out = []
    for p in parts:  # a single giant "sentence" (e.g. a schedule) still gets windowed
        if _words(p) > size:
            out += [" ".join(w) for w in _windows(p.split(), size, overlap)]
        else:
            out.append(p)
    return out


def _split_table(rows: list[list[str]], size: int) -> list[list[list[str]]]:
    header, body = rows[0], rows[1:]
    groups, current = [], []
    for row in body:
        candidate = [header] + current + [row]
        if current and _words(table_to_markdown(candidate)) > size:
            groups.append([header] + current)
            current = []
        current.append(row)
    if current or not groups:
        groups.append([header] + current)
    return groups


def structured_chunks(pages: list[Page], doc: Document, size: int, overlap: int) -> list[Chunk]:
    builder = _Builder(doc, "structured")
    heads: dict[int, str] = {}
    buf: list[tuple[str, int]] = []           # (paragraph, page)
    merged = False                            # buf spans more than one sub-section

    def section() -> str:
        levels = [k for k in sorted(heads) if not merged or k == 1]
        return " > ".join(heads[k] for k in levels)

    def flush() -> None:
        nonlocal merged
        if buf:
            builder.emit("\n\n".join(t for t, _ in buf), "text", section(), buf[0][1], buf[-1][1])
            buf.clear()
        merged = False

    for page in pages:
        for block in page.blocks:
            if block.kind == "heading":
                level = block.level or 2
                short = _words(" ".join(t for t, _ in buf)) < MIN_SECTION_WORDS
                # Never merge across a top-level part: that would mislabel the citation.
                if level == 1 or not short:
                    flush()
                elif buf:  # short sub-section: carry it forward under the shared parent
                    buf.append((block.text, page.page_no))
                    merged = True
                heads[level] = block.text
                for deeper in [k for k in heads if k > level]:
                    del heads[deeper]
            elif block.kind == "table" and block.rows:
                flush()
                for group in _split_table(block.rows, size):
                    builder.emit(table_to_markdown(group), "table", section(),
                                 page.page_no, page.page_no)
            else:
                pieces = _split_long(block.text, size, overlap) if _words(block.text) > size \
                    else [block.text]
                for piece in pieces:
                    if buf and _words(" ".join(t for t, _ in buf)) + _words(piece) > size:
                        flush()
                    buf.append((piece, page.page_no))
    flush()
    return builder.chunks


def chunk_pages(
    pages: list[Page], doc: Document, strategy: str, size: int, overlap: int
) -> list[Chunk]:
    if strategy == "naive":
        return naive_chunks(pages, doc, size, overlap)
    if strategy == "structured":
        return structured_chunks(pages, doc, size, overlap)
    raise ValueError(f"unknown chunking strategy {strategy!r}")
