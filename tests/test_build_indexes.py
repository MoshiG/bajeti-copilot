import shutil

import pytest

from bajeti.config import config_from_dict
from bajeti.corpus import Manifest
from bajeti.index.budget_db import run_readonly
from bajeti.index.build import Indexes, build_indexes
from bajeti.index.dense import HashingEmbedder, StaleIndexError
from bajeti.ingest.pipeline import ingest_corpus


@pytest.fixture
def built(tmp_path, speech_pdf, fixture_doc):
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(speech_pdf, raw / fixture_doc.filename)
    cfg = config_from_dict({"paths": {
        "raw_dir": str(raw), "processed_dir": str(tmp_path / "proc"),
        "index_dir": str(tmp_path / "idx"), "sqlite_path": str(tmp_path / "idx" / "b.sqlite"),
    }, "retrieval": {"embedding_model": "hashing"}})
    manifest = Manifest(documents=[fixture_doc])
    ingest_corpus(cfg, manifest)
    report = build_indexes(cfg, manifest, HashingEmbedder())
    return cfg, report


def test_build_reports_both_strategies_and_db(built):
    _, report = built
    assert report["embedding_model"] == "hashing-512"
    assert {s: r["status"] for s, r in report["dense"].items()} == {
        "naive": "ok", "structured": "ok"}
    assert report["budget_db"]["allocation_rows"] == 3


def test_load_and_search_end_to_end(built):
    cfg, _ = built
    idx = Indexes.load(cfg, HashingEmbedder())
    top_bm25 = idx.store[idx.bm25.search("Wizara ya Afya", k=1)[0][0]]
    assert top_bm25.kind == "table" and top_bm25.page_start == 2
    assert idx.dense.search("Sheria ya Kodi ya Mapato", k=1)[0][0]


def test_fixture_table_lands_in_sqlite_with_units(built):
    cfg, _ = built
    rows = run_readonly(cfg.paths.sqlite_path,
                        "SELECT vote_code, vote_name, total_tzs FROM allocations "
                        "ORDER BY vote_code").rows
    assert rows[0] == ("46", "Wizara ya Elimu", 1.25e12)
    docs = run_readonly(cfg.paths.sqlite_path, "SELECT doc_id, language FROM documents").rows
    assert docs == [("speech_fixture", "sw")]


def test_ablation_switches_skip_unused_indexes(built):
    cfg, _ = built
    cfg.retrieval.use_dense = False
    idx = Indexes.load(cfg)               # no embedder needed when dense is off
    assert idx.dense is None and idx.bm25 is not None


def test_reingest_without_reindex_is_caught(built, tmp_path):
    cfg, _ = built
    path = tmp_path / "proc" / "chunks_structured.jsonl"
    path.write_text(path.read_text() + "\n")   # any change to the chunks file
    with pytest.raises(StaleIndexError):
        Indexes.load(cfg, HashingEmbedder())
