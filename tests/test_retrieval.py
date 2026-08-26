"""Retrieval spot checks — the Day-5 exit criterion: top-3 hits the
correct clause for the queries the agent will actually generate.

BM25-only tests always run; hybrid tests run when the dense model is
available (skipped gracefully otherwise — CI runs lexical-only).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.retrieval import ClauseRetriever


@pytest.fixture(scope="module")
def bm25():
    return ClauseRetriever(use_dense=False)


def ids(hits):
    return [h["id"] for h in hits]


def texts(hits):
    return " ".join(h["text"].lower() for h in hits)


def test_confirmed_cancellation_charge_finds_rule6(bm25):
    hits = bm25.search("cancellation charge confirmed ticket hours before departure", k=3)
    assert "2015/rule-6" in ids(hits)


def test_vb_sleeper_query_finds_jan2026_tiers(bm25):
    hits = bm25.search("Vande Bharat Sleeper seventy-two hours cancellation charge", k=3)
    assert any(i.startswith("jan2026/") for i in ids(hits))


def test_clerkage_query_surfaces_clerkage_text(bm25):
    hits = bm25.search("RAC waitlisted ticket clerkage thirty minutes refund", k=3)
    assert "clerkage" in texts(hits)


def test_train_cancelled_query(bm25):
    hits = bm25.search("train cancelled by railways full refund", k=3)
    assert "cancel" in texts(hits) and "refund" in texts(hits)


def test_regime_boost_prefers_matching_regime(bm25):
    plain = ids(bm25.search("cancellation charge fifty per cent", k=3))
    boosted = ids(bm25.search("cancellation charge fifty per cent",
                              k=3, regime="jan2026"))
    assert any(i.startswith("jan2026/") for i in boosted)
    # the boost must not eliminate cross-regime hits entirely (soft filter)
    assert len(set(plain) & set(boosted)) >= 1


def test_by_id_binds_calculator_citations(bm25):
    for cid in ["2015/rule-6", "jan2026/6(4)(a)", "jan2026/6(4)(b)", "jan2026/6(4)(c)"]:
        c = bm25.by_id(cid)
        assert c is not None and len(c["text"]) > 50


def test_hybrid_when_dense_available():
    r = ClauseRetriever(use_dense=True)
    if not r.dense_available:
        pytest.skip("no dense backend could load (torch-less env without model2vec)")
    # a paraphrase with little lexical overlap — dense should still find tiers
    hits = r.search("how much money do I lose if I call off my trip late", k=5)
    assert hits[0]["retrievers"] == "bm25+dense"
    assert any("cancellation" in h["text"].lower() for h in hits)
