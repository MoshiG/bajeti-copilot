"""Load chunks written by ingestion and look them up by id."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bajeti.ingest.chunk import Chunk


def chunks_path(processed_dir: str | Path, strategy: str) -> Path:
    return Path(processed_dir) / f"chunks_{strategy}.jsonl"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ChunkStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        if len(self._by_id) != len(chunks):
            raise ValueError("duplicate chunk ids")

    @classmethod
    def load(cls, path: str | Path) -> ChunkStore:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found; run `make ingest` first")
        with open(path, encoding="utf-8") as fh:
            return cls([Chunk(**json.loads(line)) for line in fh if line.strip()])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    @property
    def ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]
