import shutil

import yaml

from bajeti import cli


def test_fetch_then_ingest_end_to_end(tmp_path, speech_pdf, capsys):
    raw, out, idx = tmp_path / "raw", tmp_path / "out", tmp_path / "idx"
    raw.mkdir()
    shutil.copy(speech_pdf, raw / "speech_fixture.pdf")   # simulate an already-downloaded file
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"paths": {
        "raw_dir": str(raw), "processed_dir": str(out), "index_dir": str(idx),
        "sqlite_path": str(idx / "budget.sqlite")}}))
    manifest = tmp_path / "corpus.yaml"
    manifest.write_text(yaml.safe_dump({"documents": [{
        "id": "speech_fixture", "title": "Hotuba 2025/26", "doc_type": "speech",
        "fiscal_year": "2025/26", "language": "sw", "url": "https://example.invalid/x.pdf",
    }]}))

    assert cli.main(["-c", str(cfg), "fetch", "--manifest", str(manifest)]) == 0
    assert "present" in capsys.readouterr().out
    assert (raw / "manifest.lock.json").exists()

    assert cli.main(["-c", str(cfg), "ingest", "--manifest", str(manifest)]) == 0
    report = capsys.readouterr().out
    assert "speech_fixture" in report and "ok" in report
    assert (out / "chunks_structured.jsonl").stat().st_size > 0

    assert cli.main(["-c", str(cfg), "index", "--manifest", str(manifest),
                     "--embedder", "hashing"]) == 0
    out_text = capsys.readouterr().out
    assert "dense/structured" in out_text and "allocation rows" in out_text

    cfg_with_models = tmp_path / "cfg2.yaml"
    cfg_with_models.write_text(cfg.read_text() + yaml.safe_dump({"retrieval": {
        "embedding_model": "hashing", "reranker_model": "overlap", "candidates": 10}}))
    assert cli.main(["-c", str(cfg_with_models), "search", "Wizara ya Afya", "-k", "2"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(" 1. Hotuba 2025/26, p. 2 [table]") and "rerank=1" in printed
    assert cli.main(["-c", str(cfg_with_models), "search", "kodi", "--year", "1999/00"]) == 1
