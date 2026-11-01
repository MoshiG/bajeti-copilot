import pytest

from bajeti.ingest.parse import Block, Page, clean_text, ocr_language, table_to_markdown
from tests.conftest import HEADER, SCANNED_TEXT, TABLE, has_tesseract


def _headings(pages):
    return [(b.text, b.level) for p in pages for b in p.blocks if b.kind == "heading"]


def test_page_numbers_are_one_based_and_complete(parsed_pages):
    assert [p.page_no for p in parsed_pages] == [1, 2, 3, 4]


def test_headings_detected_with_levels(parsed_pages):
    heads = _headings(parsed_pages)
    assert ("SEHEMU YA KWANZA: MAPATO", 1) in heads
    assert ("SEHEMU YA PILI: MATUMIZI", 1) in heads
    assert ("Mapato ya Kodi", 2) in heads


def test_running_header_and_page_numbers_removed(parsed_pages):
    texts = [b.text for p in parsed_pages for b in p.blocks]
    assert HEADER not in texts
    assert not any(t.strip() in {"1", "2", "3"} for t in texts)


def test_table_extracted_intact(parsed_pages):
    tables = [b for p in parsed_pages for b in p.blocks if b.kind == "table"]
    assert len(tables) == 1
    assert tables[0].rows == TABLE
    assert "| 46 | Wizara ya Elimu | 1,250,000 |" in tables[0].text


def test_table_numbers_not_duplicated_in_running_text(parsed_pages):
    paras = " ".join(b.text for p in parsed_pages for b in p.blocks if b.kind == "para")
    assert "1,250,000" not in paras


def test_reading_order_heading_before_table(parsed_pages):
    kinds = [b.kind for b in parsed_pages[1].blocks]
    assert kinds.index("heading") < kinds.index("table")


@pytest.mark.skipif(not has_tesseract(), reason="tesseract not installed")
def test_scanned_page_is_ocrd(parsed_pages):
    scanned = parsed_pages[3]
    assert scanned.ocr is True
    assert "revenue collection" in scanned.text.lower()
    assert not any(p.ocr for p in parsed_pages[:3])
    assert SCANNED_TEXT  # documents what the image contains


def test_clean_text_joins_hyphenation_and_whitespace():
    assert clean_text("mapa-\nto  ya\n kodi­") == "mapato ya kodi"


def test_markdown_table_pads_and_escapes():
    md = table_to_markdown([["a", "b"], ["x|y"]])
    assert md.splitlines() == ["| a | b |", "|---|---|", "| x/y |  |"]


def test_ocr_language_falls_back_without_swahili():
    assert ocr_language("sw", {"eng", "swa"}) == "swa+eng"
    assert ocr_language("sw", {"eng"}) == "eng"
    assert ocr_language("en", {"eng", "swa"}) == "eng"


def test_page_roundtrips_through_dict():
    page = Page("d", 3, [Block("table", "t", rows=[["a"], ["b"]]), Block("heading", "H", 1)])
    again = Page.from_dict(page.to_dict())
    assert again.page_no == 3 and again.blocks[0].rows == [["a"], ["b"]]
    assert again.blocks[1].level == 1
