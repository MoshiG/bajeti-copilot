"""Dense vector index.

Embeddings are computed once (the step worth renting a GPU for) and saved as a NumPy matrix next
to the chunk ids and a metadata file. Search is a matrix-vector product: exact, dependency-free
and fast at this scale. A vector database only pays off at millions of chunks.

The index records which chunk file and which model produced it, and refuses to load if either
has changed, so stale vectors can never silently feed an evaluation run.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from bajeti.index.bm25 import tokenize
from bajeti.index.store import ChunkStore


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    """Real embeddings (default BAAI/bge-m3, multilingual, handles Swahili and English)."""

    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, only when used

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                  show_progress_bar=len(texts) > 100, convert_to_numpy=True)
        return vecs.astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words vectors: no download, no GPU. For tests and smoke runs only."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                h = zlib.crc32(tok.encode("utf-8"))
                out[row, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return _normalize(out)


def build_embedder(name: str) -> Embedder:
    if m := re.fullmatch(r"hashing(?:-(\d+))?", name):
        return HashingEmbedder(int(m.group(1) or 512))
    return SentenceTransformerEmbedder(name)


class StaleIndexError(RuntimeError):
    pass


class DenseIndex:
    def __init__(self, ids: list[str], vectors: np.ndarray, embedder: Embedder) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors differ in length")
        self.ids, self.vectors, self.embedder = ids, vectors, embedder

    @classmethod
    def build(cls, store: ChunkStore, embedder: Embedder) -> DenseIndex:
        return cls(store.ids, embedder.encode(store.texts), embedder)

    def save(self, directory: str | Path, chunks_sha256: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.vectors)
        (d / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")
        meta = {"model": self.embedder.name, "dim": int(self.vectors.shape[1]),
                "n": len(self.ids), "chunks_sha256": chunks_sha256}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder, chunks_sha256: str) -> DenseIndex:
        d = Path(directory)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"no dense index in {d}; run `make index`")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["model"] != embedder.name:
            raise StaleIndexError(f"index built with {meta['model']}, not {embedder.name}")
        if meta["chunks_sha256"] != chunks_sha256:
            raise StaleIndexError("chunks changed since the index was built; run `make index`")
        ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
        return cls(ids, np.load(d / "embeddings.npy"), embedder)

    def search(self, query: str, k: int = 10,
               allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-k chunks by cosine similarity; `allowed` restricts ranking to a subset."""
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q
        if allowed is not None:
            mask = np.array([cid in allowed for cid in self.ids])
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]
