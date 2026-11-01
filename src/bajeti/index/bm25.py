"""BM25 keyword index.

Dense embeddings capture meaning; BM25 catches the exact strings budget questions hinge on:
vote numbers ("Fungu 46"), Act names, fiscal years ("2025/26") and amounts. The tokenizer keeps
those intact and also adds a comma-free form of every number, so "1,250,000" in a table matches
"1250000" typed by a user.

The index is rebuilt in memory from the chunk file at load time: on a corpus of a few thousand
chunks this takes well under a second, so there is nothing to persist or keep in sync.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from bajeti.index.store import ChunkStore

TOKEN = re.compile(r"\d+(?:[.,/]\d+)*|[^\W\d_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in TOKEN.findall(text.lower()):
        tokens.append(tok)
        if tok[0].isdigit() and "," in tok:
            tokens.append(tok.replace(",", ""))
    return tokens


class BM25Index:
    def __init__(self, store: ChunkStore) -> None:
        if not len(store):
            raise ValueError("cannot build BM25 over an empty chunk store")
        self.store = store
        self._ids = store.ids
        self._bm25 = BM25Okapi([tokenize(t) for t in store.texts])

    def search(self, query: str, k: int = 10,
               allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-k chunks by BM25; `allowed` restricts ranking to a subset (e.g. one fiscal year)."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        candidates = range(len(scores)) if allowed is None else [
            i for i, cid in enumerate(self._ids) if cid in allowed]
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
        return [(self._ids[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]
