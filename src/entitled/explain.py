"""LLM explanation, gated by a numeric-faithfulness validator.

The Day-6 template is the ground truth. The LLM may rephrase it — warmer,
clearer — but every numeric token in its output must already exist in the
allowed set harvested from the deterministic template (all of whose
numbers are calculator outputs by construction) plus the regime's rule
constants. One unknown number -> the LLM text is killed and the template
ships. False kills are cheap; a false pass is the failure mode.
"""

from __future__ import annotations

import re
from typing import Callable

from .agent import Answer
from .calculator import TicketCase, FLAT_2015, CLERKAGE_RESERVED
from .llm import complete

NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# rule constants that are legitimate in prose even when the template
# happens not to spell them out (tier hours, percentages, cutoffs)
REGIME_CONSTANTS = {
    "R2015":   {"48", "12", "4", "25", "50"},
    "VB2026":  {"72", "8", "25", "50"},
    "APR2026": {"72", "24", "8", "25", "50"},
}
COMMON_CONSTANTS = {"30", "60", "3", "10", str(CLERKAGE_RESERVED), "2015", "2026"}


def _norm(tok: str) -> str:
    tok = tok.replace(",", "")
    if "." in tok:
        tok = tok.rstrip("0").rstrip(".")
    return tok


def _numbers(text: str) -> set[str]:
    return {_norm(t) for t in NUM_RE.findall(text)}


def allowed_numbers(ans: Answer, case: TicketCase | None) -> set[str]:
    allowed = _numbers(ans.explanation)          # template = calculator truth
    allowed |= REGIME_CONSTANTS.get(ans.regime, set()) | COMMON_CONSTANTS
    if case is not None:
        allowed |= {str(FLAT_2015.get(case.cls, ""))} - {""}
        for dt in (case.departure, case.cancellation):
            if dt:
                allowed |= {str(dt.year), str(dt.month), str(dt.day),
                            str(dt.hour), str(dt.minute),
                            f"{dt.month:02d}", f"{dt.day:02d}",
                            f"{dt.hour:02d}", f"{dt.minute:02d}"}
    return allowed


def validate_prose(text: str, ans: Answer,
                   case: TicketCase | None) -> tuple[bool, list[str]]:
    """(ok, violations). Checks: no foreign numbers; required anchors."""
    violations = []
    foreign = _numbers(text) - allowed_numbers(ans, case)
    if foreign:
        violations.append(f"numbers not in the calculator's output: "
                          f"{sorted(foreign)}")
    low = text.lower()
    if ans.outcome == "COMPUTED":
        if f"₹{int(ans.refund):,}" not in text:
            violations.append(f"missing the exact refund figure ₹{int(ans.refund):,}")
    elif ans.outcome == "NO_REFUND":
        if "no refund" not in low:
            violations.append("missing the 'no refund' outcome statement")
    elif ans.outcome == "ESCALATE":
        if "manual review" not in low and "escalat" not in low:
            violations.append("missing the escalation statement")
    for c in ans.citations:
        if c.bound and c.clause_id not in text:
            violations.append(f"missing citation {c.clause_id}")
    if ans.outcome != "ESCALATE" and not ans.verified \
            and "provisional" not in low and "pending" not in low:
        violations.append("missing the unverified-rule caution")
    return (not violations), violations


SYSTEM = """You rewrite a railway-refund answer to be warm, clear, and short
(under 150 words). HARD RULES: keep every rupee figure, clause id (like
2015/rule-6 or jan2026/6(4)(a)), and outcome EXACTLY as given; never add,
change, or compute any number; never drop a caution. Output plain text."""


def polish(ans: Answer, case: TicketCase | None,
           llm: Callable[[str, str], tuple] = complete) -> tuple[str, str, dict]:
    """Returns (text, mode, diagnostics); mode in llm|template|kill-switch.
    NEEDS_INFO answers are never polished — clarifying questions must
    reach the user verbatim."""
    if ans.outcome == "NEEDS_INFO":
        return ans.explanation, "template", {"reason": "NEEDS_INFO not polished"}
    text, diag = llm(f"Rewrite this answer:\n\n{ans.explanation}", SYSTEM)
    if text is None:
        return ans.explanation, "template", diag
    ok, violations = validate_prose(text, ans, case)
    diag["violations"] = violations
    if not ok:
        return ans.explanation, "kill-switch", diag
    return text.strip(), "llm", diag
