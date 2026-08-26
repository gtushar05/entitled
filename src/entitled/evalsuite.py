"""Eval runner over the frozen golden set.

Two modes:
- score_offline(): structured payloads -> answer_payload. Deterministic,
  no network. Measures the spine: outcome/refund/regime/citation/
  escalation correctness. This is the number that must stay at 100%.
- score_extraction(llm): NL question -> extract -> answer, comparing the
  END-TO-END outcome to the golden expectation. Provider-dependent, so
  reported separately and never frozen.

The headline safety metric is `dangerous`: cases where the golden answer
is ESCALATE/NEEDS_INFO (the model should refuse to compute) but the agent
returned a rupee figure anyway. For a refund agent that must equal zero.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .agent import answer_payload, Answer
from .assistant import answer_question

ROOT = Path(__file__).resolve().parents[2]   # src/entitled/ -> project root
GOLDEN = ROOT / "data" / "golden.json"
COMPUTED_OUTCOMES = {"COMPUTED", "NO_REFUND"}   # produced a rupee decision


def load_golden() -> list[dict]:
    return json.loads(GOLDEN.read_text())["cases"]


def _bound_ids(ans: Answer) -> set:
    return {c.clause_id for c in ans.citations if c.bound}


def _grade(ans: Answer, e: dict) -> dict:
    """Per-field correctness for one case."""
    checks = {"outcome": ans.outcome == e["outcome"]}
    if e["refund"] is not None:
        checks["refund"] = ans.refund == e["refund"]
    if e["charge"] is not None:
        checks["charge"] = ans.charge == e["charge"]
    if e["regime"] is not None:
        checks["regime"] = ans.regime == e["regime"]
    if e["verified"] is not None:
        checks["verified"] = ans.verified == e["verified"]
    if e["must_cite"]:
        checks["citation"] = set(e["must_cite"]).issubset(_bound_ids(ans))
    return checks


def score_offline(cases: list[dict] | None = None) -> dict:
    cases = cases or load_golden()
    per_case, cat_tot, cat_ok = [], Counter(), Counter()
    trap_tot = trap_ok = plain_tot = plain_ok = 0
    refund_tot = refund_ok = cite_tot = cite_ok = 0
    dangerous = []
    confusion = defaultdict(Counter)   # expected -> actual

    for c in cases:
        ans = answer_payload(c["payload"])
        e = c["expect"]
        checks = _grade(ans, e)
        ok = all(checks.values())
        per_case.append({"id": c["id"], "ok": ok, "checks": checks,
                         "expected": e["outcome"], "actual": ans.outcome})
        cat_tot[c["category"]] += 1
        cat_ok[c["category"]] += ok
        if c["trap"]:
            trap_tot += 1; trap_ok += ok
        else:
            plain_tot += 1; plain_ok += ok
        if e["refund"] is not None:
            refund_tot += 1; refund_ok += checks.get("refund", False)
        if e["must_cite"]:
            cite_tot += 1; cite_ok += checks.get("citation", False)
        confusion[e["outcome"]][ans.outcome] += 1
        # dangerous: golden says don't compute, agent computed a number
        if e["outcome"] not in COMPUTED_OUTCOMES \
                and ans.outcome in COMPUTED_OUTCOMES:
            dangerous.append(c["id"])

    n = len(cases)
    passed = sum(p["ok"] for p in per_case)
    return {
        "n": n, "passed": passed, "exact_match": round(passed / n, 4),
        "trap_accuracy": round(trap_ok / trap_tot, 4) if trap_tot else None,
        "plain_accuracy": round(plain_ok / plain_tot, 4) if plain_tot else None,
        "refund_exactness": round(refund_ok / refund_tot, 4) if refund_tot else None,
        "citation_accuracy": round(cite_ok / cite_tot, 4) if cite_tot else None,
        "dangerous_compute_count": len(dangerous),
        "dangerous_ids": dangerous,
        "by_category": {k: {"passed": cat_ok[k], "total": cat_tot[k]}
                        for k in sorted(cat_tot)},
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "failures": [p for p in per_case if not p["ok"]],
    }


def score_extraction(llm=None, cases: list[dict] | None = None,
                     now: datetime | None = None) -> dict:
    """End-to-end from the NL question. Only the OUTCOME is graded (the
    extractor may legitimately normalize wording); refund is graded when
    the outcome matched and a number was expected. Provider-dependent."""
    cases = cases or load_golden()
    now = now or datetime(2026, 3, 18, 12, 0)
    kwargs = {"llm": llm} if llm is not None else {}
    outcome_ok = refund_ok = refund_tot = no_extract = 0
    per_case = []
    for c in cases:
        out = answer_question(c["question"], now=now, **kwargs)
        ans = out["answer"]
        actual = out["mode"] if ans is None else ans.outcome
        exp = c["expect"]["outcome"]
        o_ok = ans is not None and ans.outcome == exp
        outcome_ok += o_ok
        if ans is None:
            no_extract += 1
        if o_ok and c["expect"]["refund"] is not None:
            refund_tot += 1
            refund_ok += ans.refund == c["expect"]["refund"]
        per_case.append({"id": c["id"], "expected": exp, "actual": actual,
                         "mode": out["mode"]})
    n = len(cases)
    return {
        "n": n, "outcome_accuracy": round(outcome_ok / n, 4),
        "refund_exactness_given_outcome":
            round(refund_ok / refund_tot, 4) if refund_tot else None,
        "no_extraction": no_extract,
        "per_case": per_case,
    }


def format_report(r: dict) -> str:
    lines = [
        "Entitled — offline golden-set evaluation",
        "=" * 44,
        f"cases              {r['n']}",
        f"exact match        {r['passed']}/{r['n']}  ({r['exact_match']:.1%})",
        f"  traps            {r['trap_accuracy']:.1%}" if r['trap_accuracy'] is not None else "",
        f"  non-traps        {r['plain_accuracy']:.1%}" if r['plain_accuracy'] is not None else "",
        f"refund exactness   {r['refund_exactness']:.1%}" if r['refund_exactness'] is not None else "",
        f"citation accuracy  {r['citation_accuracy']:.1%}" if r['citation_accuracy'] is not None else "",
        f"dangerous computes {r['dangerous_compute_count']}  (must be 0)",
        "",
        "by category:",
    ]
    for k, v in r["by_category"].items():
        lines.append(f"  {k:<16} {v['passed']}/{v['total']}")
    if r["failures"]:
        lines += ["", "FAILURES:"]
        for f in r["failures"]:
            bad = [k for k, ok in f["checks"].items() if not ok]
            lines.append(f"  {f['id']}: {bad} "
                         f"(exp {f['expected']} / got {f['actual']})")
    return "\n".join(x for x in lines if x != "")
