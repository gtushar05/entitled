"""Day-11 ablations — do the moving parts earn their place?

Part A (calculator necessity): the project's central claim is that an LLM
must NOT do the arithmetic. This measures the counterfactual — give a
model the exact case facts AND the governing clauses, but no calculator,
and ask it to compute the refund itself. Whatever it scores on the trap
cases (which the verified system gets 100%) is the cost of trusting an
LLM with money. Provider-dependent.

Part B (retrieval hybrid): does BM25+dense RRF beat BM25-only / dense-only
at putting the correct clause in the top 3? Fully offline (embeddings are
local). On a 31-clause legal corpus BM25 is already strong; the honest
question is whether the dense layer adds robustness on paraphrases or is
just decoration — the numbers decide, not the intuition.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .llm import complete
from .retrieval import ClauseRetriever

# ---------------------------------------------------------------- Part A
LLM_ONLY_SYSTEM = """You are a railway refund calculator. You are given the
passenger's situation, a reference 'now' time, and the exact governing rule
clauses. Compute the outcome and refund YOURSELF from the clauses — there is
no calculator tool. Work out the hours before departure, the applicable
tier, any flat-minimum charge, and the refund.

Output ONLY a JSON object: {"outcome": "COMPUTED"|"NO_REFUND"|"ESCALATE",
"refund": <whole rupees as an integer, or null>}. Use ESCALATE only if the
clauses genuinely do not determine the answer. No other text."""


def _find_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def llm_only_answer(case: dict, retriever: ClauseRetriever,
                    llm: Callable = complete) -> dict:
    """One LLM call, clauses in context, no calculator. Returns
    {outcome, refund, raw} — refund None if unparseable/absent."""
    clauses = retriever.search(case["question"], k=5)
    ctx = "\n".join(f"[{c['id']}] {c['text']}" for c in clauses)
    prompt = (f"Reference time (now): {case['now']}\n"
              f"Situation: {case['question']}\n\n"
              f"Governing clauses:\n{ctx}\n\nJSON:")
    text, _ = llm(prompt, LLM_ONLY_SYSTEM)
    obj = _find_json(text) if text else None
    if not obj:
        return {"outcome": None, "refund": None, "raw": text}
    ref = obj.get("refund")
    return {"outcome": obj.get("outcome"),
            "refund": int(ref) if isinstance(ref, (int, float)) else None,
            "raw": text}


def score_llm_only(cases: list[dict], retriever: ClauseRetriever | None = None,
                   llm: Callable = complete) -> dict:
    """Score the no-calculator baseline. Only computational cases
    (COMPUTED/NO_REFUND/ESCALATE) — intake NEEDS_INFO is not the LLM's job
    here. `unsafe` = produced a rupee figure the rules don't support:
    wrong amount on a COMPUTED case, or any number where the answer should
    be NO_REFUND/ESCALATE."""
    retriever = retriever or ClauseRetriever(use_dense=False)
    comp = [c for c in cases
            if c["expect"]["outcome"] in ("COMPUTED", "NO_REFUND", "ESCALATE")]
    n = len(comp)
    outcome_ok = full_ok = trap_tot = trap_ok = unsafe = 0
    per = []
    for c in comp:
        e = c["expect"]
        a = llm_only_answer(c, retriever, llm)
        o_ok = a["outcome"] == e["outcome"]
        r_ok = (e["refund"] is None) or (a["refund"] == e["refund"])
        both = o_ok and r_ok
        outcome_ok += o_ok
        full_ok += both
        if c["trap"]:
            trap_tot += 1; trap_ok += both
        # unsafe: a rupee number the rules don't support
        gave_number = a["refund"] is not None and a["refund"] > 0
        if e["outcome"] in ("NO_REFUND", "ESCALATE") and gave_number:
            unsafe += 1
        elif e["outcome"] == "COMPUTED" and a["refund"] is not None \
                and a["refund"] != e["refund"]:
            unsafe += 1
        per.append({"id": c["id"], "trap": c["trap"], "outcome_ok": o_ok,
                    "full_ok": both, "exp_outcome": e["outcome"],
                    "got_outcome": a["outcome"], "exp_refund": e["refund"],
                    "got_refund": a["refund"]})
    return {
        "n": n,
        "outcome_accuracy": round(outcome_ok / n, 4) if n else None,
        "full_accuracy": round(full_ok / n, 4) if n else None,
        "trap_accuracy": round(trap_ok / trap_tot, 4) if trap_tot else None,
        "trap_n": trap_tot,
        "unsafe_rupee_outputs": unsafe,
        "per_case": per,
    }


# ---------------------------------------------------------------- Part B
# labeled retrieval queries -> (acceptable target clause id set, difficulty).
# legal text overlaps, so any clause in the set counts as a top-k hit.
# 'lexical' queries use clause terminology (what the agent actually
# generates); 'paraphrase' queries are low-lexical-overlap stress tests
# that isolate whether the dense layer earns its place.
RETRIEVAL_QUERIES = [
    ("cancellation charge on a confirmed ticket cancelled 30 hours before departure",
     {"2015/rule-6"}, "lexical"),
    ("Vande Bharat Sleeper cancelled seventy-two hours before, twenty-five per cent charge",
     {"jan2026/6(4)(a)"}, "lexical"),
    ("Vande Bharat cancelled thirty-six hours before, fifty per cent charge",
     {"jan2026/6(4)(b)"}, "lexical"),
    ("confirmed Vande Bharat ticket cancelled less than eight hours before departure",
     {"jan2026/6(4)(c)"}, "lexical"),
    ("train cancelled by the railways, full refund of fare",
     {"2015/rule-9"}, "lexical"),
    ("waitlisted or RAC ticket refund after deducting clerkage",
     {"2015/rule-4", "2015/rule-5"}, "lexical"),
    ("how much money do I lose if I call off my trip the day before the train",
     {"2015/rule-6"}, "paraphrase"),
    ("what will I get back if I scrap a waitlisted seat at the last minute",
     {"2015/rule-4", "2015/rule-5"}, "paraphrase"),
]


def retrieval_ablation() -> dict:
    r = ClauseRetriever(use_dense=True)
    n = len(RETRIEVAL_QUERIES)
    n_lex = sum(1 for *_, d in RETRIEVAL_QUERIES if d == "lexical")
    n_par = n - n_lex
    out = {"dense_available": r.dense_available, "dense_backend": r.dense_backend,
           "n_queries": n, "n_lexical": n_lex, "n_paraphrase": n_par, "modes": {}}
    for mode in ("bm25", "dense", "hybrid"):
        hits = lex = par = 0
        per = []
        for q, target, diff in RETRIEVAL_QUERIES:
            top = [h["id"] for h in r.search(q, k=3, mode=mode)]
            hit = bool(set(top) & target)
            hits += hit
            lex += hit and diff == "lexical"
            par += hit and diff == "paraphrase"
            per.append({"query": q[:48], "difficulty": diff,
                        "target": sorted(target), "top3": top, "hit": hit})
        out["modes"][mode] = {
            "recall_at_3": round(hits / n, 4), "hits": hits,
            "lexical_recall": round(lex / n_lex, 4) if n_lex else None,
            "paraphrase_recall": round(par / n_par, 4) if n_par else None,
            "per": per}
    return out
