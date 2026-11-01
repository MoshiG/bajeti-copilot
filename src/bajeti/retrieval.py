"""Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion, then reranked.

    query ─┬─> BM25  (top `candidates`) ─┐
           │                             ├─> RRF fusion ─> reranker ─> top_k hits
           └─> dense (top `candidates`) ─┘

Each stage is switched by config, which is what the ablation rows toggle:
    use_bm25 / use_dense    which retrievers run (one alone skips fusion)
    use_reranker            cross-encoder rescoring of the fused candidates

Why RRF: BM25 scores (unbounded) and cosine similarities (-1..1) are not comparable, so adding
them needs per-corpus tuning. RRF uses only rank positions: score = sum over retrievers of
1 / (k + rank). It is robust, has one parameter, and rewards chunks both retrievers agree on.

Every hit records its rank in each stage, so evaluation can show where a relevant chunk was
lost: never retrieved, retrieved but fused too low, or dropped by the reranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from bajeti.config import Config, RetrievalConfig
from bajeti.index.bm25 import tokenize
from bajeti.index.build import Indexes
from bajeti.index.dense import Embedder
from bajeti.ingest.chunk import Chunk

RRF_K = 60   # the constant from Cormack et al. (2009); larger values flatten rank differences


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly; much more precise than either retriever alone."""

    def __init__(self, model_name: str, batch_size: int = 16, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder  # heavy import, only when used

        self.name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, max_length=512, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                                     show_progress_bar=False)
        return [float(s) for s in scores]


class OverlapReranker:
    """Query-term coverage. No model: for tests and CPU smoke runs, never for reported results."""

    name = "overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(tokenize(t))) / len(q) for t in texts]


def build_reranker(name: str) -> Reranker:
    return OverlapReranker() if name == "overlap" else CrossEncoderReranker(name)


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    # Ties broken by first appearance, so results are deterministic run to run.
    order = {cid: i for i, cid in enumerate(c for r in rankings for c in r)}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


@dataclass
class Hit:
    chunk: Chunk
    score: float                                     # final score used for ordering
    ranks: dict[str, int] = field(default_factory=dict)   # 1-based rank per stage

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class Filters:
    fiscal_year: str | None = None
    doc_type: str | None = None
    language: str | None = None

    def active(self) -> bool:
        return any(v for v in (self.fiscal_year, self.doc_type, self.language))

    def allows(self, c: Chunk) -> bool:
        return all([
            not self.fiscal_year or c.fiscal_year == self.fiscal_year,
            not self.doc_type or c.doc_type == self.doc_type,
            not self.language or c.language == self.language,
        ])


class Retriever:
    def __init__(self, indexes: Indexes, cfg: RetrievalConfig,
                 reranker: Reranker | None = None) -> None:
        if cfg.use_bm25 and indexes.bm25 is None:
            raise ValueError("config enables BM25 but no BM25 index was loaded")
        if cfg.use_dense and indexes.dense is None:
            raise ValueError("config enables dense retrieval but no dense index was loaded")
        if cfg.use_reranker and reranker is None:
            raise ValueError("config enables the reranker but none was provided")
        self.indexes, self.cfg, self.reranker = indexes, cfg, reranker

    def search(self, query: str, top_k: int | None = None,
               filters: Filters | None = None) -> list[Hit]:
        query = query.strip()
        if not query:
            return []
        top_k = top_k or self.cfg.top_k
        n = max(self.cfg.candidates, top_k)
        allowed = None
        if filters and filters.active():
            allowed = {c.chunk_id for c in self.indexes.store.chunks if filters.allows(c)}
            if not allowed:
                return []

        stages: dict[str, list[str]] = {}
        if self.cfg.use_bm25:
            stages["bm25"] = [cid for cid, _ in self.indexes.bm25.search(query, n, allowed)]
        if self.cfg.use_dense:
            stages["dense"] = [cid for cid, _ in self.indexes.dense.search(query, n, allowed)]

        fused = rrf_fuse(list(stages.values()))[:n]
        if len(stages) > 1:
            stages["fused"] = [cid for cid, _ in fused]
        ranks = {cid: {name: ranking.index(cid) + 1 for name, ranking in stages.items()
                       if cid in ranking} for cid, _ in fused}

        hits = [Hit(self.indexes.store[cid], score, ranks[cid]) for cid, score in fused]
        if self.cfg.use_reranker and hits:
            scores = self.reranker.score(query, [h.chunk.text for h in hits])
            for h, s in zip(hits, scores, strict=True):
                h.score = s
            hits.sort(key=lambda h: -h.score)        # stable: ties keep fused order
            for i, h in enumerate(hits, start=1):
                h.ranks["rerank"] = i
        return hits[:top_k]


def build_retriever(cfg: Config, embedder: Embedder | None = None,
                    reranker: Reranker | None = None) -> Retriever:
    indexes = Indexes.load(cfg, embedder)
    if cfg.retrieval.use_reranker and reranker is None:
        reranker = build_reranker(cfg.retrieval.reranker_model)
    return Retriever(indexes, cfg.retrieval, reranker)


def format_hits(hits: list[Hit], max_chars: int = 1800) -> str:
    """Numbered context blocks for the LLM: [n] title, pages, section, then the text."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        body = c.text.split("\n", 1)[1] if c.text.startswith("[") and "\n" in c.text else c.text
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " …"
        where = f" — {c.section}" if c.section else ""
        blocks.append(f"[{i}] {c.citation}{where}\n{body}")
    return "\n\n".join(blocks)
