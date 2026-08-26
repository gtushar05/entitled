"""Citation-grounded retrieval over the clause store.

Hybrid search: BM25 (lexical) + MiniLM embeddings (semantic), fused by
reciprocal-rank fusion — no score-scale juggling, robust at tiny corpus
size. Every hit carries its clause id, source, and regime tag, so the
agent's answers cite clauses, not page blobs.

Role boundary (repeated from calculator.py because it matters): retrieval
EXPLAINS and CITES; it never computes. A wrong retrieval can produce a
wrong citation, never a wrong rupee amount.

The dense layer is a backend chain: sentence-transformers (torch) where
available, else model2vec static embeddings (pure NumPy — the working path
on Python 3.14 / Intel Mac, where torch has no wheels), else BM25-only
with the degradation surfaced via `dense_available` — the demo's
kill-switch pattern, applied to search.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
CLAUSES = ROOT / "corpus" / "parsed" / "clauses.json"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _emb_cache(backend: str) -> Path:
    # per-backend cache: embedding dims differ (MiniLM 384 vs potion 256),
    # so a cache written by one backend must never be served to another
    slug = re.sub(r"[^a-z0-9]+", "_", backend.lower())
    return ROOT / "corpus" / "parsed" / f"clause_emb_{slug}.npy"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class ClauseRetriever:
    def __init__(self, use_dense: bool = True):
        store = json.loads(CLAUSES.read_text())
        self.clauses = store["clauses"]
        self._bm25 = BM25Okapi([_tokens(c["text"]) for c in self.clauses])
        self._encode = None
        self._emb = None
        self.dense_available = False
        self.dense_backend = None
        if use_dense:
            self._init_dense()

    def _init_dense(self):
        """Backend chain: sentence-transformers (torch) -> model2vec (pure
        NumPy static embeddings; the working path on Python 3.14 / Intel Mac,
        where torch has no wheels) -> BM25-only degradation."""
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(DENSE_MODEL)
            self._encode = lambda texts: m.encode(texts, normalize_embeddings=True)
            self.dense_backend = "sentence-transformers/MiniLM"
        except Exception:
            try:
                from model2vec import StaticModel
                m = StaticModel.from_pretrained("minishlab/potion-base-8M")

                def enc(texts):
                    e = np.asarray(m.encode(texts))
                    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
                self._encode = enc
                self.dense_backend = "model2vec/potion-base-8M"
            except Exception:
                return  # BM25-only; surfaced via dense_available
        cache = _emb_cache(self.dense_backend)
        emb = None
        if cache.exists():
            cached = np.load(cache)
            if cached.shape[0] == len(self.clauses):
                emb = cached
        if emb is None:
            emb = self._encode([c["text"] for c in self.clauses])
            np.save(cache, emb)
        self._emb = emb
        self.dense_available = True

    # ---- ranking primitives ----
    def _bm25_ranks(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(_tokens(query))
        return list(np.argsort(-scores))

    def _dense_ranks(self, query: str) -> list[int]:
        q = self._encode([query])[0]
        sims = self._emb @ q
        return list(np.argsort(-sims))

    def search(self, query: str, k: int = 5,
               regime: str | None = None) -> list[dict]:
        """Reciprocal-rank fusion of the available rankers.

        RRF score = sum over rankers of 1/(60 + rank). Optionally boosts
        clauses whose regime tag matches the case's regime (soft filter:
        other regimes remain retrievable for 'what changed?' questions).
        """
        rankings = [self._bm25_ranks(query)]
        if self.dense_available:
            rankings.append(self._dense_ranks(query))
        rrf = np.zeros(len(self.clauses))
        for ranks in rankings:
            for pos, idx in enumerate(ranks):
                rrf[idx] += 1.0 / (60 + pos)
        if regime:
            for i, c in enumerate(self.clauses):
                if c["regime"].startswith(regime):
                    rrf[i] *= 1.5
        order = np.argsort(-rrf)[:k]
        return [{**self.clauses[i],
                 "score": float(rrf[i]),
                 "retrievers": "bm25+dense" if self.dense_available else "bm25-only"}
                for i in order]

    def by_id(self, clause_id: str) -> dict | None:
        """Exact lookup — how the calculator's citation ids bind to text."""
        for c in self.clauses:
            if c["id"] == clause_id:
                return c
        return None
