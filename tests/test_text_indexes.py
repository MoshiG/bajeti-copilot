import numpy as np
import pytest

from bajeti.index.bm25 import BM25Index, tokenize
from bajeti.index.dense import DenseIndex, HashingEmbedder, StaleIndexError, build_embedder
from bajeti.index.store import ChunkStore
from bajeti.ingest.chunk import Chunk


def _chunk(i, text):
    return Chunk(chunk_id=f"d:s{i:04d}", doc_id="d", title="T", doc_type="speech",
                 fiscal_year="2025/26", language="sw", strategy="structured", kind="text",
                 section="", page_start=1, page_end=1, text=text, n_words=len(text.split()))


@pytest.fixture
def store():
    return ChunkStore([
        _chunk(0, "Fungu 46 Wizara ya Elimu imetengewa shilingi 1,250,000 milioni mwaka 2025/26"),
        _chunk(1, "Serikali inapendekeza marekebisho ya Sheria ya Kodi ya Mapato"),
        _chunk(2, "Health sector support project receives foreign financing"),
        _chunk(3, "Mheshimiwa Spika, naomba kutoa hoja"),
    ])


def test_tokenize_keeps_codes_and_adds_comma_free_numbers():
    toks = tokenize("Fungu 46: TZS 1,250,000 kwa 2025/26.")
    assert toks == ["fungu", "46", "tzs", "1,250,000", "1250000", "kwa", "2025/26"]


def test_bm25_finds_exact_vote_and_amount(store):
    bm25 = BM25Index(store)
    assert bm25.search("Fungu 46", k=1)[0][0] == "d:s0000"
    assert bm25.search("1250000", k=1)[0][0] == "d:s0000"   # matches "1,250,000"
    assert bm25.search("kodi ya mapato", k=1)[0][0] == "d:s0001"
    assert bm25.search("???") == []


def test_bm25_rejects_empty_store():
    with pytest.raises(ValueError):
        BM25Index(ChunkStore([]))


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=64)
    a, b = e.encode(["kodi ya mapato"]), e.encode(["kodi ya mapato"])
    assert np.array_equal(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert e.encode([""])[0].sum() == 0


def test_build_embedder_resolves_hashing_names():
    assert build_embedder("hashing").dim == 512
    assert build_embedder("hashing-128").dim == 128


def test_dense_search_and_roundtrip(store, tmp_path):
    e = HashingEmbedder()
    index = DenseIndex.build(store, e)
    assert index.search("health project foreign financing", k=1)[0][0] == "d:s0002"
    index.save(tmp_path, "sha-1")
    again = DenseIndex.load(tmp_path, e, "sha-1")
    assert again.ids == index.ids and np.allclose(again.vectors, index.vectors)


def test_stale_index_refused(store, tmp_path):
    DenseIndex.build(store, HashingEmbedder()).save(tmp_path, "sha-1")
    with pytest.raises(StaleIndexError, match="chunks changed"):
        DenseIndex.load(tmp_path, HashingEmbedder(), "sha-2")
    with pytest.raises(StaleIndexError, match="built with"):
        DenseIndex.load(tmp_path, HashingEmbedder(dim=64), "sha-1")
    with pytest.raises(FileNotFoundError):
        DenseIndex.load(tmp_path / "missing", HashingEmbedder(), "sha-1")


def test_chunk_store_lookup_and_duplicates(store):
    assert store["d:s0001"].text.startswith("Serikali")
    with pytest.raises(ValueError):
        ChunkStore([_chunk(0, "a"), _chunk(0, "b")])
    with pytest.raises(FileNotFoundError, match="make ingest"):
        ChunkStore.load("/nonexistent/chunks.jsonl")
