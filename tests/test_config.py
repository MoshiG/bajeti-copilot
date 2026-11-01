from pathlib import Path

import pytest

from bajeti.config import config_from_dict, deep_merge, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "default.yaml"
ABLATIONS = sorted((ROOT / "configs" / "ablations").glob("*.yaml"))


def test_default_config_loads():
    cfg = load_config(DEFAULT)
    assert cfg.retrieval.use_bm25 and cfg.retrieval.use_dense
    assert cfg.llm.temperature == 0.0


def test_deep_merge_overrides_only_given_keys():
    base = {"retrieval": {"use_bm25": True, "top_k": 6}, "llm": {"model": "a"}}
    out = deep_merge(base, {"retrieval": {"use_bm25": False}})
    assert out == {"retrieval": {"use_bm25": False, "top_k": 6}, "llm": {"model": "a"}}
    assert base["retrieval"]["use_bm25"] is True  # base untouched


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown key"):
        config_from_dict({"retrieval": {"use_bm2": True}})


def test_invalid_combinations_are_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"retrieval": {"use_bm25": False, "use_dense": False}})
    with pytest.raises(ValueError):
        config_from_dict({"ingest": {"chunking": "semantic"}})


def test_env_overrides_llm(monkeypatch):
    monkeypatch.setenv("BAJETI_LLM_BASE_URL", "http://gpu-pod:8000/v1")
    monkeypatch.setenv("BAJETI_LLM_MODEL", "some/model")
    cfg = load_config(DEFAULT)
    assert cfg.llm.base_url == "http://gpu-pod:8000/v1"
    assert cfg.llm.model == "some/model"


def test_there_are_five_ablation_configs():
    assert len(ABLATIONS) == 5


@pytest.mark.parametrize("path", ABLATIONS, ids=lambda p: p.stem)
def test_every_ablation_overlay_is_valid(path):
    cfg = load_config(DEFAULT, path)
    assert cfg.name == path.stem
