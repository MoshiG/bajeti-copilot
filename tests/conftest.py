"""Shared fixtures: a small synthetic budget-speech PDF that exercises every parsing path."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

HEADER = "Hotuba ya Bajeti Kuu ya Serikali 2025/26"
PARA_REVENUE = (
    "1. Mheshimiwa Spika, makusanyo ya mapato ya ndani yameendelea kuimarika katika mwaka "
    "huu wa fedha kutokana na hatua za kuboresha usimamizi wa kodi, matumizi ya mifumo ya "
    "kielektroniki na kupanua wigo wa walipakodi katika sekta zote za uchumi wa nchi yetu."
)
PARA_TAX = (
    "2. Mheshimiwa Spika, Serikali inapendekeza kufanya marekebisho katika Sheria ya Kodi ya "
    "Mapato ili kuongeza uwazi, kurahisisha ulipaji wa kodi na kuchochea uwekezaji katika "
    "viwanda vidogo na vya kati pamoja na kilimo cha kibiashara."
)
PARA_SPEND = (
    "3. Mheshimiwa Spika, mgawanyo wa fedha za maendeleo kwa mafungu makuu umeonyeshwa katika "
    "jedwali lifuatalo kwa mwaka wa fedha 2025/26."
)
TABLE = [
    ["Fungu", "Taasisi", "Kiasi (TZS milioni)"],
    ["46", "Wizara ya Elimu", "1,250,000"],
    ["52", "Wizara ya Afya", "980,500"],
    ["98", "Wizara ya Ujenzi", "2,100,000"],
]
PARA_CLOSE = (
    "4. Mheshimiwa Spika, naomba kutoa hoja kwamba Bunge lako tukufu lijadili na kupitisha "
    "makadirio ya mapato na matumizi ya Serikali kwa mwaka wa fedha 2025/26."
)
SCANNED_TEXT = "Revenue collection improved strongly this year"


def _build_pdf(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()

    def page_with_frame(n: int):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 40), HEADER, fontsize=9, fontname="helv")
        page.insert_text((290, 815), str(n), fontsize=9, fontname="helv")
        return page

    p1 = page_with_frame(1)
    p1.insert_text((72, 90), "SEHEMU YA KWANZA: MAPATO", fontsize=16, fontname="hebo")
    p1.insert_textbox(pymupdf.Rect(72, 105, 523, 200), PARA_REVENUE, fontsize=11, fontname="helv")
    p1.insert_text((72, 230), "Mapato ya Kodi", fontsize=13, fontname="hebo")
    p1.insert_textbox(pymupdf.Rect(72, 245, 523, 340), PARA_TAX, fontsize=11, fontname="helv")

    p2 = page_with_frame(2)
    p2.insert_text((72, 90), "SEHEMU YA PILI: MATUMIZI", fontsize=16, fontname="hebo")
    p2.insert_textbox(pymupdf.Rect(72, 105, 523, 170), PARA_SPEND, fontsize=11, fontname="helv")
    x = [72, 152, 382, 523]
    y0, row_h = 190, 24
    for r, row in enumerate(TABLE):
        for c, cell in enumerate(row):
            p2.insert_text((x[c] + 6, y0 + r * row_h + 16), cell, fontsize=10, fontname="helv")
    for r in range(len(TABLE) + 1):
        p2.draw_line((x[0], y0 + r * row_h), (x[-1], y0 + r * row_h), width=0.8)
    for xc in x:
        p2.draw_line((xc, y0), (xc, y0 + len(TABLE) * row_h), width=0.8)

    p3 = page_with_frame(3)
    p3.insert_textbox(pymupdf.Rect(72, 80, 523, 200), PARA_CLOSE, fontsize=11, fontname="helv")

    # Page 4: an image-only "scan" with no text layer.
    scratch = pymupdf.open()
    s = scratch.new_page(width=595, height=200)
    s.insert_text((40, 100), SCANNED_TEXT, fontsize=22, fontname="helv")
    pix = s.get_pixmap(dpi=200)
    p4 = doc.new_page(width=595, height=842)
    p4.insert_image(pymupdf.Rect(0, 100, 595, 300), pixmap=pix)

    doc.save(path)


@pytest.fixture(scope="session")
def speech_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdfs") / "speech_fixture.pdf"
    _build_pdf(path)
    return path


@pytest.fixture(scope="session")
def parsed_pages(speech_pdf):
    from bajeti.ingest.parse import parse_pdf

    return parse_pdf(speech_pdf, "speech_fixture", language="sw", ocr_fallback=True)


@pytest.fixture
def fixture_doc():
    from bajeti.corpus import Document

    return Document(id="speech_fixture", title="Hotuba ya Bajeti 2025/26", doc_type="speech",
                    fiscal_year="2025/26", language="sw", manual=True)


@pytest.fixture(scope="session")
def built_cfg(tmp_path_factory, speech_pdf):
    """Fixture PDF ingested and indexed (hashing vectors, overlap reranker): a full mini corpus."""
    from bajeti.config import config_from_dict
    from bajeti.corpus import Document, Manifest
    from bajeti.index.build import build_indexes
    from bajeti.index.dense import HashingEmbedder
    from bajeti.ingest.pipeline import ingest_corpus

    root = tmp_path_factory.mktemp("built")
    (root / "raw").mkdir()
    shutil.copy(speech_pdf, root / "raw" / "speech_fixture.pdf")
    cfg = config_from_dict({
        "paths": {"raw_dir": str(root / "raw"), "processed_dir": str(root / "p"),
                  "index_dir": str(root / "i"), "sqlite_path": str(root / "i" / "b.sqlite")},
        "retrieval": {"embedding_model": "hashing", "reranker_model": "overlap",
                      "candidates": 10, "top_k": 3},
    })
    doc = Document(id="speech_fixture", title="Hotuba ya Bajeti 2025/26", doc_type="speech",
                   fiscal_year="2025/26", language="sw", manual=True)
    manifest = Manifest(documents=[doc])
    ingest_corpus(cfg, manifest)
    build_indexes(cfg, manifest, HashingEmbedder())
    return cfg


@pytest.fixture
def retriever(built_cfg):
    from bajeti.index.dense import HashingEmbedder
    from bajeti.retrieval import OverlapReranker, build_retriever

    return build_retriever(built_cfg, HashingEmbedder(), OverlapReranker())


def has_tesseract() -> bool:
    return shutil.which("tesseract") is not None
