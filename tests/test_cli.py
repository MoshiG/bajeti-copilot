import json
from pathlib import Path

from bajeti import cli
from bajeti.llm import FakeLLM

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = str(ROOT / "configs" / "default.yaml")
A1 = str(ROOT / "configs" / "ablations" / "a1_naive_dense.yaml")


def test_config_command_layers_overlays(capsys):
    assert cli.main(["-c", DEFAULT, "-c", A1, "config"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["name"] == "a1_naive_dense"
    assert out["retrieval"]["use_bm25"] is False
    assert out["ingest"]["chunking"] == "naive"


def test_ping_uses_configured_llm(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_llm", lambda cfg: FakeLLM(["tayari"]))
    assert cli.main(["-c", DEFAULT, "ping"]) == 0
    assert "'tayari'" in capsys.readouterr().out
