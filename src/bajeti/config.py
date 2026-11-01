"""Typed configuration loaded from YAML, with overlays for ablation runs.

Usage:
    cfg = load_config("configs/default.yaml")                      # base
    cfg = load_config("configs/default.yaml", "configs/ablations/a3_hybrid_rerank.yaml")
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key_env: str = "BAJETI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: int = 120

    @property
    def api_key(self) -> str:
        # vLLM accepts any non-empty key; hosted APIs need the real one.
        return os.environ.get(self.api_key_env, "EMPTY")


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    index_dir: str = "data/index"
    sqlite_path: str = "data/index/budget.sqlite"
    gold_set: str = "eval/gold.jsonl"
    results_dir: str = "eval/results"


@dataclass
class IngestConfig:
    chunking: str = "structured"
    chunk_tokens: int = 400
    chunk_overlap: int = 60
    ocr_fallback: bool = True

    def __post_init__(self) -> None:
        if self.chunking not in {"naive", "structured"}:
            raise ValueError(
                f"ingest.chunking must be 'naive' or 'structured', got {self.chunking!r}"
            )
        if not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError("ingest.chunk_overlap must be >= 0 and smaller than chunk_tokens")


@dataclass
class RetrievalConfig:
    use_bm25: bool = True
    use_dense: bool = True
    use_reranker: bool = True
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    candidates: int = 30
    top_k: int = 6

    def __post_init__(self) -> None:
        if not (self.use_bm25 or self.use_dense):
            raise ValueError("retrieval needs at least one of use_bm25 / use_dense")
        if self.top_k > self.candidates:
            raise ValueError("retrieval.top_k cannot exceed retrieval.candidates")


@dataclass
class ToolsConfig:
    enable_sql: bool = True
    enable_optimizer: bool = True


@dataclass
class AgentConfig:
    max_steps: int = 5
    require_citations: bool = True


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    judge: LLMConfig | None = None          # evaluation-time judge; optional
    name: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "llm": LLMConfig,
    "paths": PathsConfig,
    "ingest": IngestConfig,
    "retrieval": RetrievalConfig,
    "tools": ToolsConfig,
    "agent": AgentConfig,
}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where overlay's keys replace base's, recursing into nested dicts."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _env_overrides() -> dict[str, Any]:
    """Allow BAJETI_LLM_BASE_URL / BAJETI_LLM_MODEL to override the YAML (handy on a rented GPU)."""
    llm: dict[str, Any] = {}
    if url := os.environ.get("BAJETI_LLM_BASE_URL"):
        llm["base_url"] = url
    if model := os.environ.get("BAJETI_LLM_MODEL"):
        llm["model"] = model
    return {"llm": llm} if llm else {}


def config_from_dict(data: dict[str, Any]) -> Config:
    kwargs: dict[str, Any] = {}
    for section, cls in _SECTIONS.items():
        raw = data.get(section) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown key(s) in '{section}': {sorted(unknown)}")
        kwargs[section] = cls(**raw)
    unknown_top = set(data) - set(_SECTIONS) - {"name", "judge"}
    if unknown_top:
        raise ValueError(f"Unknown top-level section(s): {sorted(unknown_top)}")
    if data.get("judge"):
        raw = data["judge"]
        unknown = set(raw) - {f.name for f in fields(LLMConfig)}
        if unknown:
            raise ValueError(f"Unknown key(s) in 'judge': {sorted(unknown)}")
        kwargs["judge"] = LLMConfig(**raw)
    return Config(name=data.get("name", "default"), **kwargs)


def load_config(*paths: str | Path) -> Config:
    """Load one or more YAML files, later files overriding earlier ones, then env overrides."""
    if not paths:
        raise ValueError("load_config needs at least one YAML path")
    merged: dict[str, Any] = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            merged = deep_merge(merged, yaml.safe_load(fh) or {})
    merged = deep_merge(merged, _env_overrides())
    return config_from_dict(merged)
