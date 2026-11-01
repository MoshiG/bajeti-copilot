import json
import shutil

import pytest

from bajeti.config import config_from_dict
from bajeti.corpus import Manifest
from bajeti.ingest import pipeline
from bajeti.ingest.pipeline import ingest_corpus, read_jsonl


@pytest.fixture
def workspace(tmp_path, speech_pdf, fixture_doc):
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(speech_pdf, raw / fixture_doc.filename)
    cfg = config_from_dict({"paths": {"raw_dir": str(raw), "processed_dir": str(tmp_path / "out")}})
    missing = fixture_doc.__class__(id="not_downloaded", title="X", doc_type="finance_act",
                                    fiscal_year="2025/26", language="en", url="https://x")
    return cfg, Manifest(documents=[fixture_doc, missing]), tmp_path / "out"


def test_ingest_writes_all_outputs(workspace):
    cfg, manifest, out = workspace
    reports = {r.doc_id: r for r in ingest_corpus(cfg, manifest)}

    ok = reports["speech_fixture"]
    assert ok.status == "ok" and ok.pages == 4 and ok.tables == 1
    assert ok.chunks["naive"] > 0 and ok.chunks["structured"] > 0
    assert reports["not_downloaded"].status == "missing"

    for name in ["chunks_naive.jsonl", "chunks_structured.jsonl", "tables.jsonl"]:
        assert (out / name).exists()
    table = read_jsonl(out / "tables.jsonl")[0]
    assert table["page_no"] == 2 and table["rows"][1][0] == "46"
    assert json.loads((out / "ingest_report.json").read_text())[0]["doc_id"] == "speech_fixture"


def test_second_run_uses_page_cache(workspace, monkeypatch):
    cfg, manifest, out = workspace
    ingest_corpus(cfg, manifest)

    def boom(*a, **k):
        raise AssertionError("parse_pdf should not run when pages are cached")

    monkeypatch.setattr(pipeline, "parse_pdf", boom)
    reports = ingest_corpus(cfg, manifest)
    assert reports[0].status == "ok"


def test_partial_run_does_not_overwrite_chunk_files(workspace):
    cfg, manifest, out = workspace
    ingest_corpus(cfg, manifest)
    before = (out / "chunks_structured.jsonl").read_text()
    ingest_corpus(cfg, manifest, only={"speech_fixture"})
    assert (out / "chunks_structured.jsonl").read_text() == before


def test_parse_failure_is_reported_not_raised(workspace, monkeypatch):
    cfg, manifest, out = workspace

    def broken(*a, **k):
        raise RuntimeError("corrupt PDF")

    monkeypatch.setattr(pipeline, "parse_pdf", broken)
    reports = ingest_corpus(cfg, manifest, force=True)
    assert reports[0].status == "failed" and "corrupt" in reports[0].message
