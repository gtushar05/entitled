"""Demo logic for the Gradio Space — deliberately gradio-free so it is
unit-testable without the UI dependency. app.py is a thin wrapper.

Design for a PUBLIC, shared, budget-constrained Space:
- The hero is the DETERMINISTIC path: the structured-input form and the
  trap gallery run the calculator with NO LLM, so the demo is fully
  functional with zero provider cost and no cold starts.
- The natural-language path needs a provider (an ANTHROPIC_API_KEY set as
  a Space secret; there is no `claude` CLI on HF). It is gated by a
  global RequestBudget so a public demo cannot run up cost, and by an
  input-length cap. When no provider is configured or the budget is spent,
  it falls back to CACHED TRACES — the kill-switch pattern applied to the
  demo itself, so visitors still see real NL->answer behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent import Answer, answer_payload
from .assistant import answer_question

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "golden.json"
CACHED_TRACES = ROOT / "data" / "cached_traces.json"

MAX_INPUT_CHARS = 600


class RequestBudget:
    """Process-global cap on LLM-backed requests for the shared Space."""

    def __init__(self, max_calls: int = 40):
        self.max_calls = max_calls
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    def allow(self) -> bool:
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True


def render_answer(ans: Answer, mode: str | None = None) -> str:
    """Answer -> markdown. The explanation is already citation-grounded;
    this adds a compact status line and a faithfulness badge."""
    badge = {"llm": "✍️ LLM-rephrased, numeric-faithfulness gate PASSED",
             "kill-switch": "🛑 LLM output failed the faithfulness gate — "
                            "showing the verified template instead",
             "template": "⚙️ deterministic template (no LLM used)",
             "no-extractor": "⚙️ no language model available"}
    lines = [ans.explanation, ""]
    tag = {"COMPUTED": "🟢 COMPUTED", "NO_REFUND": "🟡 NO REFUND",
           "ESCALATE": "🔵 ESCALATE (manual review)",
           "NEEDS_INFO": "❓ NEEDS INFO"}.get(ans.outcome, ans.outcome)
    meta = f"**{tag}** · regime `{ans.regime}`"
    if ans.outcome not in ("NEEDS_INFO",):
        meta += f" · {'primary-verified' if ans.verified else 'PROVISIONAL (primary pending)'}"
    if mode:
        meta += f"\n\n_{badge.get(mode, mode)}_"
    lines.append(meta)
    return "\n".join(lines)


def answer_structured(payload: dict) -> str:
    """Deterministic path — no LLM, always available."""
    ans = answer_payload(payload)
    return render_answer(ans, mode="template" if ans.outcome != "NEEDS_INFO" else None)


def provider_available() -> bool:
    import os
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def answer_nl(question: str, budget: RequestBudget) -> tuple[str, str]:
    """Natural-language path. Returns (markdown, status). Falls back to a
    cached trace (or a disabled note) when no provider / over budget."""
    q = (question or "").strip()
    if not q:
        return "_Enter a situation above._", "idle"
    if len(q) > MAX_INPUT_CHARS:
        return (f"⚠️ Input too long ({len(q)} chars; cap {MAX_INPUT_CHARS}). "
                "Please shorten it."), "rejected"
    if not provider_available():
        cached = match_cached_trace(q)
        note = ("ℹ️ No language model is configured on this demo, so the "
                "natural-language parser is offline. Showing a **cached "
                "trace** for a similar example — the structured-input tab "
                "runs the full calculator live with no model needed.")
        return f"{note}\n\n---\n\n{cached}", "cached"
    if not budget.allow():
        return ("🛑 The shared demo's request budget for this session is "
                "spent. Use the structured-input tab (runs live, no model) "
                "or try again later."), "budget-exceeded"
    out = answer_question(q)
    body = render_answer(out["answer"], out["mode"]) if out["answer"] \
        else out["text"]
    status = f"ok ({budget.remaining} model requests left)"
    return body, status


# ---------------------------------------------------- cached traces & gallery
def _load_golden() -> list[dict]:
    return json.loads(GOLDEN.read_text())["cases"]


def build_cached_traces() -> list[dict]:
    """Pre-render answers for a spread of example questions. The payloads
    are the golden cases', so these traces are exactly what the live agent
    produces — an honest cache, not a mock."""
    picks = ["r2015-3a-36h", "r2015-3a-48h", "vb-2a-2400-72h", "apr-sl-300-36h",
             "r2015-cancelled-e-sl-900", "r2015-wl-sl-900-20min", "ni-unreserved"]
    by_id = {c["id"]: c for c in _load_golden()}
    out = []
    for cid in picks:
        c = by_id[cid]
        ans = answer_payload(c["payload"])
        out.append({"id": cid, "question": c["question"],
                    "markdown": render_answer(
                        ans, "template" if ans.outcome != "NEEDS_INFO" else None)})
    return out


def match_cached_trace(question: str) -> str:
    """Cheapest possible retrieval over cached traces: token overlap with
    the cached questions. Good enough for a fallback demonstration."""
    traces = load_cached_traces()
    if not traces:
        return "_(no cached traces available)_"
    import re
    qt = set(re.findall(r"[a-z0-9]+", question.lower()))
    best = max(traces, key=lambda t: len(
        qt & set(re.findall(r"[a-z0-9]+", t["question"].lower()))))
    return f"**Cached example:** _{best['question']}_\n\n{best['markdown']}"


def load_cached_traces() -> list[dict]:
    if CACHED_TRACES.exists():
        return json.loads(CACHED_TRACES.read_text())
    return build_cached_traces()


def trap_gallery() -> list[dict]:
    """The frozen trap cases with their verified answers — pure data, no
    compute: tier boundaries, flat-minimum binding, escalation discipline."""
    out = []
    for c in _load_golden():
        if not c["trap"]:
            continue
        ans = answer_payload(c["payload"])
        out.append({"id": c["id"], "question": c["question"],
                    "rationale": c["rationale"],
                    "markdown": render_answer(
                        ans, "template" if ans.outcome != "NEEDS_INFO" else None)})
    return out
