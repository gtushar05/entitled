"""Day-12 demo-logic tests — offline, no gradio, no provider.

The demo's guarantees under test: the deterministic path always answers;
the NL path never calls a model without budget/provider and falls back to
a cached trace; the request budget is a hard cap; cached traces are honest
(equal to the live deterministic answer)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled import demo
from entitled.demo import (RequestBudget, answer_structured, answer_nl,
                           build_cached_traces, trap_gallery, MAX_INPUT_CHARS)

GOOD = {"fare": 1500, "cls": "3A", "departure": "2026-03-20 18:00",
        "cancellation": "2026-03-19 18:00"}


# ---------------------------------------------------------------- budget
def test_budget_hard_caps():
    b = RequestBudget(max_calls=2)
    assert b.allow() and b.allow()
    assert not b.allow()               # third denied
    assert b.remaining == 0


# ------------------------------------------------------- deterministic path
def test_structured_answer_is_cited_and_needs_no_model():
    md = answer_structured(GOOD)
    assert "₹1,125" in md and "2015/rule-6" in md
    assert "no LLM used" in md.lower() or "deterministic" in md.lower()


def test_structured_needs_info_surfaces_questions():
    md = answer_structured({"cls": "3A"})
    assert "NEEDS INFO" in md and "fare" in md.lower()


# ------------------------------------------------------------- NL fallback
def test_nl_without_provider_serves_cached_trace(monkeypatch):
    monkeypatch.setattr(demo, "provider_available", lambda: False)
    b = RequestBudget(5)
    body, status = answer_nl("my 3AC ticket, cancelling a day before", b)
    assert status == "cached"
    assert "cached" in body.lower()
    assert b.used == 0                 # no model call was made


def test_nl_rejects_overlong_input(monkeypatch):
    monkeypatch.setattr(demo, "provider_available", lambda: True)
    b = RequestBudget(5)
    body, status = answer_nl("x" * (MAX_INPUT_CHARS + 1), b)
    assert status == "rejected" and b.used == 0


def test_nl_over_budget_refuses_without_calling_model(monkeypatch):
    monkeypatch.setattr(demo, "provider_available", lambda: True)
    b = RequestBudget(0)               # nothing left
    body, status = answer_nl("anything", b)
    assert status == "budget-exceeded"


def test_nl_empty_is_idle():
    body, status = answer_nl("   ", RequestBudget(5))
    assert status == "idle"


# --------------------------------------------------------------- traces
def test_cached_traces_are_honest():
    """Every cached trace equals the live deterministic answer for its
    case — a real cache, not a mock."""
    from entitled.agent import answer_payload
    from entitled.demo import render_answer
    golden = {c["id"]: c for c in json.loads(
        (Path(demo.GOLDEN)).read_text())["cases"]}
    for t in build_cached_traces():
        c = golden[t["id"]]
        ans = answer_payload(c["payload"])
        expected = render_answer(
            ans, "template" if ans.outcome != "NEEDS_INFO" else None)
        assert t["markdown"] == expected


def test_trap_gallery_covers_traps_with_answers():
    g = trap_gallery()
    assert len(g) >= 15
    ids = {t["id"] for t in g}
    assert "r2015-3a-48h" in ids       # a tier-boundary trap
    for t in g:
        assert t["markdown"] and t["rationale"]
