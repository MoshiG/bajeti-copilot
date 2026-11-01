import os
import shutil

import pytest

from bajeti.config import RetrievalConfig, config_from_dict
from bajeti.corpus import Manifest
from bajeti.index.build import Indexes, build_indexes
from bajeti.index.dense import HashingEmbedder
from bajeti.index.store import ChunkStore
from bajeti.ingest.chunk import Chunk
from bajeti.ingest.pipeline import ingest_corpus
from bajeti.retrieval import (
    Filters,
    OverlapReranker,
    Retriever,
    build_retriever,
    format_hits,
    rrf_fuse,
)


def _chunk(cid, text, year="2025/26", doc_type="speech", lang="sw"):
    return Chunk(chunk_id=cid, doc_id="d", title="Hotuba", doc_type=doc_type, fiscal_year=year,
                 language=lang, strategy="structured", kind="text", section="S",
                 page_start=3, page_end=3, text=f"[Hotuba | S]\n{text}", n_words=len(text.split()))


class StubIndex:
    """Returns a fixed ranking, honouring `allowed`, so fusion logic is tested in isolation."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def search(self, query, k, allowed=None):
        self.calls.append((query, k, allowed))
        ids = [c for c in self.ranking if allowed is None or c in allowed]
        return [(c, 1.0 / (i + 1)) for i, c in enumerate(ids[:k])]


class StubReranker:
    name = "stub"

    def __init__(self, prefer):
        self.prefer = prefer

    def score(self, query, texts):
        return [1.0 if self.prefer in t else 0.0 for t in texts]


@pytest.fixture
def store():
    return ChunkStore([
        _chunk("a", "Fungu 46 elimu"), _chunk("b", "afya na maji"),
        _chunk("c", "kodi ya mapato", year="2024/25"), _chunk("d", "tax policy", lang="en"),
        _chunk("e", "ujenzi wa barabara"),
    ])


def _retriever(store, bm25=("a", "b", "c"), dense=("b", "d", "a"), reranker=None, **cfg):
    rcfg = RetrievalConfig(**{"use_reranker": False, "candidates": 10, "top_k": 3, **cfg})
    idx = Indexes(store, StubIndex(list(bm25)) if rcfg.use_bm25 else None,
                  StubIndex(list(dense)) if rcfg.use_dense else None, None)
    return Retriever(idx, rcfg, reranker)


def test_rrf_rewards_agreement_and_is_deterministic():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [cid for cid, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    tie = rrf_fuse([["x"], ["y"]])
    assert [c for c, _ in tie] == ["x", "y"]


def test_hybrid_fuses_and_records_stage_ranks(store):
    hits = _retriever(store).search("q")
    assert [h.chunk_id for h in hits] == ["b", "a", "d"]
    assert hits[0].ranks == {"bm25": 2, "dense": 1, "fused": 1}
    assert hits[2].ranks == {"dense": 2, "fused": 3}       # never found by BM25


@pytest.mark.parametrize("flags, expected", [
    ({"use_bm25": True, "use_dense": False}, ["a", "b", "c"]),
    ({"use_bm25": False, "use_dense": True}, ["b", "d", "a"]),
])
def test_single_retriever_modes_skip_fusion(store, flags, expected):
    hits = _retriever(store, **flags).search("q")
    assert [h.chunk_id for h in hits] == expected
    assert all("fused" not in h.ranks for h in hits)


def test_reranker_reorders_and_is_recorded(store):
    hits = _retriever(store, reranker=StubReranker("barabara"), use_reranker=True,
                      bm25=("a", "b", "e"), dense=("b", "a", "e")).search("q", top_k=2)
    assert hits[0].chunk_id == "e" and hits[0].ranks["rerank"] == 1
    assert hits[0].ranks["fused"] == 3        # rescued from fused rank 3


def test_filters_restrict_before_ranking(store):
    r = _retriever(store)
    hits = r.search("q", filters=Filters(fiscal_year="2024/25"))
    assert [h.chunk_id for h in hits] == ["c"]
    _, _, allowed = r.indexes.bm25.calls[-1]
    assert allowed == {"c"}
    assert r.search("q", filters=Filters(language="en"))[0].chunk_id == "d"
    assert r.search("q", filters=Filters(fiscal_year="1999/00")) == []


def test_empty_query_and_top_k(store):
    r = _retriever(store)
    assert r.search("   ") == []
    assert len(r.search("q", top_k=1)) == 1


def test_misconfiguration_fails_fast(store):
    with pytest.raises(ValueError, match="reranker"):
        _retriever(store, use_reranker=True)
    idx = Indexes(store, None, StubIndex(["a"]), None)
    with pytest.raises(ValueError, match="BM25"):
        Retriever(idx, RetrievalConfig(use_reranker=False), None)


def test_overlap_reranker_scores_coverage():
    scores = OverlapReranker().score("kodi ya mapato", ["kodi ya mapato 2025", "afya", ""])
    assert scores == [1.0, 0.0, 0.0]
    assert OverlapReranker().score("???", ["x"]) == [0.0]


def test_format_hits_numbers_and_cites(store):
    hits = _retriever(store).search("q", top_k=2)
    text = format_hits(hits)
    assert text.startswith("[1] Hotuba, p. 3 — S\nafya na maji")
    assert "\n\n[2] Hotuba, p. 3" in text
    assert "[Hotuba | S]" not in text           # header not repeated inside the body


def test_format_hits_truncates_long_chunks(store):
    hit = _retriever(store).search("q", top_k=1)[0]
    hit.chunk.text = "[H]\n" + "neno " * 1000
    assert len(format_hits([hit], max_chars=100)) < 160


@pytest.fixture
def real_cfg(tmp_path, speech_pdf, fixture_doc):
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(speech_pdf, raw / fixture_doc.filename)
    cfg = config_from_dict({
        "paths": {"raw_dir": str(raw), "processed_dir": str(tmp_path / "p"),
                  "index_dir": str(tmp_path / "i"), "sqlite_path": str(tmp_path / "i/b.sqlite")},
        "retrieval": {"embedding_model": "hashing", "reranker_model": "overlap",
                      "candidates": 10, "top_k": 3},
    })
    ingest_corpus(cfg, Manifest(documents=[fixture_doc]))
    build_indexes(cfg, None, HashingEmbedder())
    return cfg


def test_end_to_end_on_parsed_pdf(real_cfg):
    r = build_retriever(real_cfg, HashingEmbedder())
    top = r.search("Wizara ya Afya fungu 52")[0]
    assert top.chunk.kind == "table" and top.chunk.page_start == 2
    assert set(top.ranks) == {"bm25", "dense", "fused", "rerank"}


@pytest.mark.parametrize("ablation", ["a1_naive_dense", "a2_hybrid", "a3_hybrid_rerank",
                                      "a4_structured", "a5_full"])
def test_every_ablation_config_builds_a_working_retriever(real_cfg, ablation):
    from pathlib import Path

    import yaml

    from bajeti.config import deep_merge

    overlay = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "ablations" / f"{ablation}.yaml").read_text())
    cfg = config_from_dict(deep_merge(real_cfg.to_dict(), overlay))
    hits = build_retriever(cfg, HashingEmbedder()).search("mapato ya kodi")
    assert hits and hits[0].chunk.strategy == cfg.ingest.chunking


@pytest.mark.skipif(os.environ.get("BAJETI_INTEGRATION") != "1",
                    reason="set BAJETI_INTEGRATION=1 on a machine with the models")
def test_real_models_rank_swahili_query_against_english_text():
    from bajeti.index.dense import SentenceTransformerEmbedder
    from bajeti.retrieval import CrossEncoderReranker

    texts = ["The government allocated funds to the Ministry of Health for hospitals.",
             "Road construction in Dodoma region will continue this year."]
    vecs = SentenceTransformerEmbedder("BAAI/bge-m3").encode(["fedha za afya"] + texts)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]
    scores = CrossEncoderReranker("BAAI/bge-reranker-v2-m3").score("fedha za afya", texts)
    assert scores[0] > scores[1]
