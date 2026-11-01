import pytest

from bajeti.ingest.chunk import _split_long, _split_table, chunk_pages
from bajeti.ingest.parse import Block, Page


def test_structured_chunks_follow_sections(parsed_pages, fixture_doc):
    chunks = chunk_pages(parsed_pages, fixture_doc, "structured", size=400, overlap=60)
    sections = {c.section for c in chunks}
    assert "SEHEMU YA KWANZA: MAPATO > Mapato ya Kodi" in sections
    assert all(c.text.startswith("[Hotuba ya Bajeti 2025/26") for c in chunks)


def test_structured_keeps_table_as_own_chunk(parsed_pages, fixture_doc):
    chunks = chunk_pages(parsed_pages, fixture_doc, "structured", size=400, overlap=60)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1
    assert tables[0].section == "SEHEMU YA PILI: MATUMIZI"
    assert tables[0].page_start == tables[0].page_end == 2
    assert "| 52 | Wizara ya Afya | 980,500 |" in tables[0].text


def test_structured_respects_size_limit(parsed_pages, fixture_doc):
    chunks = chunk_pages(parsed_pages, fixture_doc, "structured", size=25, overlap=5)
    assert all(c.n_words <= 25 for c in chunks if c.kind == "text")
    assert len(chunks) > 4


def test_naive_windows_overlap_and_track_pages(parsed_pages, fixture_doc):
    chunks = chunk_pages(parsed_pages, fixture_doc, "naive", size=30, overlap=10)
    assert all(c.n_words <= 30 for c in chunks)
    first, second = (c.text.split("\n", 1)[1].split() for c in chunks[:2])
    assert first[-10:] == second[:10]
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 4
    assert {c.section for c in chunks} == {""} and {c.kind for c in chunks} == {"text"}


def test_chunk_ids_unique_and_citations_readable(parsed_pages, fixture_doc):
    chunks = chunk_pages(parsed_pages, fixture_doc, "structured", size=400, overlap=60)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[0].citation.startswith("Hotuba ya Bajeti 2025/26, p")


def test_short_subsection_merges_under_shared_parent(fixture_doc):
    pages = [Page("d", 1, [
        Block("heading", "PART I", 1), Block("heading", "Intro", 2), Block("para", "Short intro."),
        Block("heading", "Details", 2), Block("para", "word " * 50),
    ])]
    chunks = chunk_pages(pages, fixture_doc, "structured", size=400, overlap=0)
    assert len(chunks) == 1 and "Short intro." in chunks[0].text
    assert chunks[0].section == "PART I"  # not "PART I > Details", which would mislabel the intro


def test_short_section_never_crosses_a_part_boundary(fixture_doc):
    pages = [Page("d", 1, [
        Block("heading", "PART I", 1), Block("para", "Only a few words here."),
        Block("heading", "PART II", 1), Block("para", "word " * 50),
    ])]
    chunks = chunk_pages(pages, fixture_doc, "structured", size=400, overlap=0)
    assert [c.section for c in chunks] == ["PART I", "PART II"]


def test_long_paragraph_splits_at_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    parts = _split_long(text, size=20, overlap=0)
    assert all(len(p.split()) <= 20 for p in parts)
    assert all(p.rstrip().endswith(".") for p in parts)


def test_table_split_repeats_header():
    rows = [["Vote", "Amount"]] + [[str(i), "1,000,000"] for i in range(40)]
    groups = _split_table(rows, size=40)
    assert len(groups) > 1 and all(g[0] == ["Vote", "Amount"] for g in groups)
    assert sum(len(g) - 1 for g in groups) == 40


def test_unknown_strategy_rejected(fixture_doc):
    with pytest.raises(ValueError):
        chunk_pages([], fixture_doc, "semantic", 100, 10)
