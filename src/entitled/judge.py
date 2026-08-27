"""LLM-as-judge for numeric/citation faithfulness, plus the agreement math.

Day-9 question: does the cheap DETERMINISTIC gate (explain.validate_prose)
agree with an independent LLM judge about whether a candidate answer is
faithful to the calculator's template? We measure this against a
GROUND-TRUTH set (prose we deliberately corrupted in known ways), so we
get three things at once:
  - validator accuracy vs ground truth (does the gate catch corruptions?)
  - judge accuracy vs ground truth (is the judge any good?)
  - Cohen's kappa between validator and judge (do they agree beyond chance?)

The judge is a SECOND line of defense, not the primary one: the validator
is deterministic and free and runs on every request; the judge is used
offline to characterize the validator's blind spots (notably the
value-swap case the token-set check cannot see).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .agent import answer_payload
from .llm import complete

_GOLDEN = Path(__file__).resolve().parents[2] / "data" / "golden.json"

JUDGE_SYSTEM = """You audit a customer-facing railway refund answer for
FAITHFULNESS to an authoritative reference answer. The reference is
correct by construction. Judge ONLY faithfulness, not style:

FAITHFUL means every rupee amount, every percentage, every clause id
(e.g. 2015/rule-6, jan2026/6(4)(a)), and the outcome (refund / no refund /
escalate) in the CANDIDATE agree with the REFERENCE, with no invented or
altered numbers and no dropped caution about provisional/pending rules.
If the candidate swaps two amounts (e.g. states the cancellation charge as
the refund), that is UNFAITHFUL.

Respond with a single JSON object: {"faithful": true|false, "reason":
"<one sentence>"}. No other text."""


def judge_faithfulness(reference: str, candidate: str,
                       llm: Callable[[str, str], tuple] = complete
                       ) -> tuple[bool | None, dict]:
    """(faithful, diag). faithful=None when no provider/unparseable — the
    caller must treat None as 'no judgment', never as a pass."""
    prompt = (f"REFERENCE (authoritative):\n{reference}\n\n"
              f"CANDIDATE (audit this):\n{candidate}\n\nJSON:")
    text, diag = llm(prompt, JUDGE_SYSTEM)
    if text is None:
        return None, diag
    import re
    import json
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        diag["raw_head"] = text[:120]
        return None, diag
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        diag["raw_head"] = text[:120]
        return None, diag
    diag["reason"] = obj.get("reason", "")
    return bool(obj.get("faithful")), diag


def _money(x: int) -> str:
    return f"₹{int(x):,}"


def build_faithfulness_samples() -> list[dict]:
    """Ground-truth prose set for the judge/validator study.

    reference = the faithful deterministic template. candidate = either the
    same (faithful) or a KNOWN corruption. `expect_validator_pass` encodes
    documented behavior: the token-set validator catches wrong numbers,
    invented percentages, dropped citations and dropped cautions, but by
    construction CANNOT see a value-swap (both amounts are already in the
    allowed set) — that miss is the whole reason a semantic judge exists.
    """
    by_id = {c["id"]: c for c in json.loads(_GOLDEN.read_text())["cases"]}

    def ans_of(cid):
        c = by_id[cid]
        a = answer_payload(c["payload"])
        return c, a

    samples: list[dict] = []

    def add(cid, kind, candidate, gt_faithful, expect_pass, ref):
        samples.append({"case_id": cid, "kind": kind, "reference": ref,
                        "candidate": candidate, "gt_faithful": gt_faithful,
                        "expect_validator_pass": expect_pass})

    # ---- corruption generators ----
    def wrong_refund(a):
        return a.explanation.replace(f"Refund due: {_money(a.refund)}",
                                     f"Refund due: {_money(a.refund + 50)}", 1)

    def invented_pct(a):
        return a.explanation + "\nThat works out to roughly 37% of your fare."

    def dropped_citation(a):
        cid = next(c.clause_id for c in a.citations if c.bound)
        return a.explanation.replace(cid, "the applicable rule")

    def dropped_caution(a):
        return "\n".join(l for l in a.explanation.splitlines()
                         if "provisional" not in l.lower()
                         and "pending" not in l.lower())

    def value_swap(a):
        ra, rc = _money(a.refund), _money(a.charge)
        t = a.explanation.replace(ra, "§A§", 1).replace(rc, "§B§", 1)
        return t.replace("§A§", rc).replace("§B§", ra)

    # representative COMPUTED cases (verified, with bound cites)
    for cid in ["r2015-3a-36h", "vb-2a-2400-102h", "r2015-sl-60h"]:
        c, a = ans_of(cid)
        ref = a.explanation
        add(cid, "faithful", ref, True, True, ref)
        add(cid, "wrong_refund", wrong_refund(a), False, False, ref)
        add(cid, "invented_percentage", invented_pct(a), False, False, ref)
        add(cid, "dropped_citation", dropped_citation(a), False, False, ref)
        add(cid, "value_swap", value_swap(a), False, True, ref)   # <- blind spot

    # unverified COMPUTED case (Apr-2026): tests the caution requirement
    c, a = ans_of("apr-3a-36h")
    ref = a.explanation
    add("apr-3a-36h", "faithful", ref, True, True, ref)
    add("apr-3a-36h", "dropped_caution", dropped_caution(a), False, False, ref)
    add("apr-3a-36h", "invented_percentage", invented_pct(a), False, False, ref)
    add("apr-3a-36h", "value_swap", value_swap(a), False, True, ref)

    # faithful non-COMPUTED anchors (balance the label distribution)
    for cid in ["r2015-3a-4h", "apr-3a-102h", "r2015-wl-pt-esc"]:
        c, a = ans_of(cid)
        add(cid, "faithful", a.explanation, True, True, a.explanation)

    return samples


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary labelers over the same items.
    kappa = (po - pe) / (1 - pe). Returns 1.0 when both are constant and
    identical (perfect but chance-undefined -> report as agreement)."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be equal, non-empty length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa_t = sum(a) / n
    pb_t = sum(b) / n
    pe = pa_t * pb_t + (1 - pa_t) * (1 - pb_t)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)
