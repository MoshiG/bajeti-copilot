"""Build every index from ingestion outputs, and load them back for querying.

data/index/
    <strategy>/embeddings.npy, ids.json, meta.json     dense vectors per chunking strategy
    budget.sqlite                                        budget tables (shared by all strategies)
    index_report.json                                    what was built, with which model
BM25 needs no files: it is rebuilt from the chunk file in memory when loaded.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bajeti.config import Config
from bajeti.corpus import Manifest
from bajeti.index.bm25 import BM25Index
from bajeti.index.budget_db import build_budget_db, load_tables
from bajeti.index.dense import DenseIndex, Embedder, build_embedder
from bajeti.index.store import ChunkStore, chunks_path, file_sha256
from bajeti.ingest.pipeline import STRATEGIES


def dense_dir(cfg: Config, strategy: str) -> Path:
    return Path(cfg.paths.index_dir) / strategy


def build_indexes(
    cfg: Config,
    manifest: Manifest | None = None,
    embedder: Embedder | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
) -> dict[str, Any]:
    embedder = embedder or build_embedder(cfg.retrieval.embedding_model)
    report: dict[str, Any] = {"embedding_model": embedder.name, "dense": {}, "budget_db": None}

    for strategy in strategies:
        path = chunks_path(cfg.paths.processed_dir, strategy)
        if not path.exists():
            report["dense"][strategy] = {"status": "skipped", "message": f"{path} missing"}
            continue
        store = ChunkStore.load(path)
        started = time.perf_counter()
        index = DenseIndex.build(store, embedder)
        index.save(dense_dir(cfg, strategy), file_sha256(path))
        report["dense"][strategy] = {
            "status": "ok", "chunks": len(store), "dim": int(index.vectors.shape[1]),
            "seconds": round(time.perf_counter() - started, 2),
        }

    tables_file = Path(cfg.paths.processed_dir) / "tables.jsonl"
    if tables_file.exists():
        documents = [{**asdict(d), "doc_id": d.id} for d in manifest.documents] if manifest else []
        est_file = Path(cfg.paths.processed_dir) / "estimates.json"
        estimates = json.loads(est_file.read_text(encoding="utf-8")) if est_file.exists() \
            else None
        report["budget_db"] = build_budget_db(load_tables(tables_file), cfg.paths.sqlite_path,
                                              documents, estimates)

    out = Path(cfg.paths.index_dir) / "index_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@dataclass
class Indexes:
    """Everything retrieval needs for one chunking strategy."""
    store: ChunkStore
    bm25: BM25Index | None
    dense: DenseIndex | None
    sqlite_path: Path

    @classmethod
    def load(cls, cfg: Config, embedder: Embedder | None = None) -> Indexes:
        strategy = cfg.ingest.chunking
        path = chunks_path(cfg.paths.processed_dir, strategy)
        store = ChunkStore.load(path)
        bm25 = BM25Index(store) if cfg.retrieval.use_bm25 else None
        dense = None
        if cfg.retrieval.use_dense:
            embedder = embedder or build_embedder(cfg.retrieval.embedding_model)
            dense = DenseIndex.load(dense_dir(cfg, strategy), embedder, file_sha256(path))
        return cls(store, bm25, dense, Path(cfg.paths.sqlite_path))
