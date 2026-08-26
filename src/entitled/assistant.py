"""End-to-end: natural-language question -> cited, validated answer.

     question ──LLM──> payload ──parser──> TicketCase ──calculator──> Result
                                                                        │
     final text <──validator-gated LLM polish <── template <── citations┘

The two LLM touchpoints (extract, polish) are both optional and both
guarded: extraction output goes through parse_case, polish output through
validate_prose. With no LLM at all this degrades to structured input +
template answers — the Day-6 spine, fully functional.

The 'Understood:' echo is deterministic and always present: the user must
be able to SEE what the extractor thought they said, because a wrong
input is the one error the downstream guards cannot catch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from .agent import answer_payload, Answer
from .extract import extract_fields
from .explain import polish
from .llm import complete
from .parser import parse_case

NO_EXTRACTOR_MSG = (
    "I could not parse the situation automatically (no language model is "
    "available). Please provide the fields directly: fare, class, quota, "
    "status, departure and cancellation date-times, train type, and any "
    "disruption.")

_ECHO_FIELDS = ["fare", "cls", "class", "quota", "status", "channel",
                "train_type", "departure", "cancellation", "disruption",
                "travelled", "chart_prepared"]


def _echo(payload: dict) -> str:
    parts = [f"{k}={payload[k]!r}" for k in _ECHO_FIELDS if k in payload]
    return "Understood: " + (", ".join(parts) if parts else "(no fields)")


def answer_question(question: str, now: datetime | None = None,
                    llm: Callable[[str, str], tuple] = complete) -> dict:
    """Returns {answer: Answer|None, text, mode, payload, diagnostics}."""
    now = now or datetime.now()
    payload, ex_diag = extract_fields(question, now, llm)
    if payload is None:
        return {"answer": None, "text": NO_EXTRACTOR_MSG,
                "mode": "no-extractor", "payload": None,
                "diagnostics": {"extract": ex_diag}}
    ans: Answer = answer_payload(payload)
    case = parse_case(payload).case          # None when NEEDS_INFO
    text, mode, po_diag = polish(ans, case, llm)
    return {"answer": ans,
            "text": f"{_echo(payload)}\n\n{text}",
            "mode": mode, "payload": payload,
            "diagnostics": {"extract": ex_diag, "polish": po_diag}}
