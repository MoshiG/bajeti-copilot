"""PDF parsing that keeps what retrieval needs: page numbers, headings, tables and scanned text.

For each page we produce an ordered list of blocks:
    heading  - detected from font size, bold weight or patterns like "PART III" / "SEHEMU YA PILI"
    para     - running text
    table    - rows extracted by pdfplumber, kept out of the running text so numbers stay aligned
Pages with no text layer are OCR'd. Running headers, footers and page numbers that repeat
across pages are removed, because they otherwise pollute every chunk and every embedding.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

HEADING_PATTERNS = [
    re.compile(r"^(PART|SEHEMU)\s+[IVXLC\d]+\b", re.I),
    re.compile(r"^SEHEMU\s+YA\s+\w+", re.I),
    re.compile(r"^(CHAPTER|SURA)\s+[IVXLC\d]+\b", re.I),
]
LEVEL1_PATTERNS = HEADING_PATTERNS[:2]
PAGE_NUMBER = re.compile(r"^\s*(page\s+|ukurasa\s+)?[\divxlc]+(\s*(of|/)\s*\d+)?\s*$", re.I)
MIN_TEXT_CHARS = 25          # below this, a page is treated as scanned and OCR'd
REPEAT_SHARE = 0.5           # a line on >50% of pages' edges is a running header/footer


@dataclass
class Block:
    kind: str                            # heading | para | table
    text: str
    level: int = 0                       # headings: 1 (part) or 2 (sub-heading)
    rows: list[list[str]] | None = None  # tables only
    top: float = 0.0                     # y position, used for reading order

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("top")
        return {k: v for k, v in d.items() if v not in (None, 0) or k == "level"}


@dataclass
class Page:
    doc_id: str
    page_no: int                         # 1-based, as printed in citations
    blocks: list[Block] = field(default_factory=list)
    ocr: bool = False

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page_no": self.page_no,
            "ocr": self.ocr,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Page:
        blocks = [Block(**b) for b in d["blocks"]]
        return cls(doc_id=d["doc_id"], page_no=d["page_no"], blocks=blocks, ocr=d.get("ocr", False))


# --- text helpers ------------------------------------------------------------------------------


def clean_text(text: str) -> str:
    text = text.replace("­", "")                     # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # words broken across lines
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def clean_cell(cell: Any) -> str:
    return re.sub(r"\s+", " ", str(cell)).strip() if cell is not None else ""


def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[c.replace("|", "/") for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(lines)


def _signature(text: str) -> str:
    """Normalise a line so 'Hotuba ya Bajeti 2025/26 — 14' and '... — 15' compare equal."""
    return re.sub(r"\d+", "#", text.lower()).strip()


# --- core parsing ------------------------------------------------------------------------------


def _inside(bbox: tuple[float, float, float, float], boxes: list[tuple]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def _extract_tables(plumber_page: Any) -> list[tuple[tuple, list[list[str]]]]:
    out = []
    for table in plumber_page.find_tables():
        rows = [[clean_cell(c) for c in row] for row in table.extract()]
        rows = [r for r in rows if any(r)]
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
            out.append((tuple(table.bbox), rows))
    return out


def _span_stats(doc: Any) -> float:
    """Body font size = the size carrying the most characters in the document."""
    sizes: Counter[float] = Counter()
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _classify(text: str, sizes: list[float], bold_share: float, body: float) -> tuple[str, int]:
    words = len(text.split())
    if any(p.match(text) for p in LEVEL1_PATTERNS):
        return "heading", 1
    if words == 0 or words > 15 or text.endswith((",", ";", ":")):
        return "para", 0
    size = max(sizes) if sizes else body
    if size >= body * 1.4:
        return "heading", 1
    if size >= body * 1.15 or (bold_share > 0.8 and not text.endswith(".")):
        return "heading", 2
    if any(p.match(text) for p in HEADING_PATTERNS) or (text.isupper() and 2 <= words <= 12):
        return "heading", 2
    return "para", 0


def _text_blocks(page: Any, table_boxes: list[tuple], body: float) -> list[Block]:
    blocks: list[Block] = []
    for raw in page.get_text("dict").get("blocks", []):
        if raw.get("type", 0) != 0 or _inside(raw["bbox"], table_boxes):
            continue
        spans = [s for line in raw.get("lines", []) for s in line.get("spans", [])]
        lines = ["".join(s["text"] for s in line.get("spans", [])) for line in raw.get("lines", [])]
        text = clean_text("\n".join(lines))
        if not text:
            continue
        chars = sum(len(s["text"].strip()) for s in spans) or 1
        bold = sum(len(s["text"].strip()) for s in spans if s.get("flags", 0) & 16) / chars
        kind, level = _classify(text, [s["size"] for s in spans if s["text"].strip()], bold, body)
        blocks.append(Block(kind=kind, text=text, level=level, top=raw["bbox"][1]))
    return blocks


def available_ocr_languages() -> set[str]:
    try:
        import pytesseract

        return set(pytesseract.get_languages(config=""))
    except Exception:
        return set()


def ocr_language(doc_language: str, available: set[str]) -> str:
    wanted = ["swa", "eng"] if doc_language == "sw" else ["eng"]
    have = [lang for lang in wanted if lang in available]
    if "swa" in wanted and "swa" not in have:
        log.warning("Swahili OCR pack missing (apt install tesseract-ocr-swa); using English")
    return "+".join(have) or "eng"


def _ocr_page(page: Any, lang: str, dpi: int = 300) -> list[Block]:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    raw = pytesseract.image_to_string(image, lang=lang)
    paras = [clean_text(p) for p in re.split(r"\n\s*\n", raw)]
    return [Block(kind="para", text=p, top=i) for i, p in enumerate(paras) if p]


def _drop_running_lines(pages: list[Page]) -> None:
    """Remove page numbers and headers/footers repeated on the edges of many pages."""
    if len(pages) >= 3:
        edge: Counter[str] = Counter()
        for p in pages:
            text_blocks = [b for b in p.blocks if b.kind != "table"]
            for b in text_blocks[:2] + text_blocks[-2:]:
                edge[_signature(b.text)] += 1
        repeated = {s for s, n in edge.items() if n / len(pages) > REPEAT_SHARE}
    else:
        repeated = set()
    for p in pages:
        p.blocks = [
            b for b in p.blocks
            if b.kind == "table"
            or not (PAGE_NUMBER.match(b.text) or _signature(b.text) in repeated)
        ]


def parse_pdf(
    path: str | Path,
    doc_id: str,
    language: str = "en",
    ocr_fallback: bool = True,
    extract_tables: bool = True,
) -> list[Page]:
    import pdfplumber
    import pymupdf

    pages: list[Page] = []
    ocr_lang: str | None = None
    with pymupdf.open(path) as doc, pdfplumber.open(path) as plumber:
        body = _span_stats(doc)
        for i, page in enumerate(doc):
            tables = _extract_tables(plumber.pages[i]) if extract_tables else []
            boxes = [bbox for bbox, _ in tables]
            blocks = _text_blocks(page, boxes, body)
            blocks += [
                Block(kind="table", text=table_to_markdown(rows), rows=rows, top=bbox[1])
                for bbox, rows in tables
            ]
            blocks.sort(key=lambda b: b.top)

            is_ocr = False
            if sum(len(b.text) for b in blocks) < MIN_TEXT_CHARS and ocr_fallback:
                ocr_lang = ocr_lang or ocr_language(language, available_ocr_languages())
                try:
                    blocks, is_ocr = _ocr_page(page, ocr_lang), True
                except Exception as exc:  # missing binary or a corrupt image should not stop a run
                    log.warning("OCR failed on %s p.%d: %s", doc_id, i + 1, exc)
            pages.append(Page(doc_id=doc_id, page_no=i + 1, blocks=blocks, ocr=is_ocr))

    _drop_running_lines(pages)
    return pages
