"""Day-11 ablation tests — offline (retrieval is local; LLM-only uses fakes)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.ablation import (retrieval_ablation, RETRIEVAL_QUERIES,
                               llm_only_answer, score_llm_only)
from entitled.retrieval import ClauseRetriever

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = {c["id"]: c for c in json.loads((ROOT / "data" / "golden.json").read_text())["cases"]}


# ------------------------------------------------------- Part B retrieval
def test_retrieval_ablation_structure():
    r = retrieval_ablation()
    assert set(r["modes"]) == {"bm25", "dense", "hybrid"}
    for mode in r["modes"].values():
        assert 0.0 <= mode["recall_at_3"] <= 1.0
        assert mode["hits"] == sum(p["hit"] for p in mode["per"])


def test_lexical_queries_mostly_reachable_by_hybrid():
    # the realistic case: clause-terminology queries the agent generates
    r = retrieval_ablation()
    assert r["modes"]["hybrid"]["lexical_recall"] >= 0.8
    assert r["modes"]["bm25"]["lexical_recall"] >= 0.8


def test_dense_earns_place_on_paraphrases():
    # the honest ablation result: dense catches at least one low-overlap
    # paraphrase that lexical BM25 misses entirely
    r = retrieval_ablation()
    assert r["modes"]["dense"]["paraphrase_recall"] > r["modes"]["bm25"]["paraphrase_recall"]


def test_hybrid_no_worse_than_bm25_overall():
    r = retrieval_ablation()
    assert r["modes"]["hybrid"]["recall_at_3"] >= r["modes"]["bm25"]["recall_at_3"]


def test_search_mode_selects_rankers():
    r = ClauseRetriever(use_dense=True)
    q = "confirmed ticket cancellation charge"
    for mode in ("bm25", "dense", "hybrid"):
        hits = r.search(q, k=3, mode=mode)
        assert len(hits) == 3


# ------------------------------------------------------- Part A llm-only
def fake_llm(outcome, refund):
    reply = json.dumps({"outcome": outcome, "refund": refund})
    return lambda prompt, system=None: (reply, {"provider": "fake"})


def test_llm_only_answer_parses():
    c = GOLDEN["r2015-3a-36h"]
    a = llm_only_answer(c, ClauseRetriever(use_dense=False),
                        llm=fake_llm("COMPUTED", 1125))
    assert a["outcome"] == "COMPUTED" and a["refund"] == 1125


def test_llm_only_answer_handles_garbage():
    c = GOLDEN["r2015-3a-36h"]
    a = llm_only_answer(c, ClauseRetriever(use_dense=False),
                        llm=lambda p, s=None: ("no json", {}))
    assert a["outcome"] is None and a["refund"] is None


def test_score_counts_unsafe_fabrication():
    # golden apr-3a-102h is ESCALATE; an LLM that returns a rupee number is unsafe
    esc = GOLDEN["apr-3a-102h"]
    r = score_llm_only([esc], ClauseRetriever(use_dense=False),
                       llm=fake_llm("COMPUTED", 1125))
    assert r["unsafe_rupee_outputs"] == 1
    assert r["full_accuracy"] == 0.0


def test_score_counts_wrong_amount_as_unsafe():
    comp = GOLDEN["r2015-3a-36h"]      # COMPUTED, refund 1125
    r = score_llm_only([comp], ClauseRetriever(use_dense=False),
                       llm=fake_llm("COMPUTED", 1300))   # wrong number
    assert r["unsafe_rupee_outputs"] == 1
    assert r["outcome_accuracy"] == 1.0    # outcome right, amount wrong
    assert r["full_accuracy"] == 0.0


def test_score_perfect_llm_is_safe():
    comp = GOLDEN["r2015-3a-36h"]
    r = score_llm_only([comp], ClauseRetriever(use_dense=False),
                       llm=fake_llm("COMPUTED", 1125))
    assert r["unsafe_rupee_outputs"] == 0 and r["full_accuracy"] == 1.0


def test_score_skips_needs_info_cases():
    # NEEDS_INFO golden cases aren't the LLM-only baseline's job
    r = score_llm_only([GOLDEN["ni-unreserved"]], ClauseRetriever(use_dense=False),
                       llm=fake_llm("COMPUTED", 40))
    assert r["n"] == 0
