"""Provider cost / latency / quality benchmark — the 'which model do we
ship, at what cost' frontier.

For a given model tier it runs the full assistant pipeline (extract ->
compute -> polish) over a sample of golden cases, instruments every LLM
call for latency and token volume, and scores two qualities that matter
for THIS system:
  - extraction accuracy: did NL -> structured -> outcome land correctly?
  - polish faithfulness rate: of the answers the LLM rephrased, how many
    passed the numeric-faithfulness gate (vs were killed to template)?
Cost is computed from token volume x published per-MTok pricing.

Honesty notes:
- The deterministic spine's correctness does NOT depend on the model —
  the calculator is the same regardless. What the model tier buys is
  extraction quality and prose faithfulness, at a latency/cost.
- Token counts here are ESTIMATES (chars/4) unless the API `usage` field
  is available; the report labels which. Latency is measured wall-clock.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

from .assistant import answer_question

# per-MTok (input, output) USD. Source: claude-api skill pricing table,
# cached 2026-06-24. Keyed by CLI tier alias.
PRICING = {"haiku": (1.00, 5.00), "sonnet": (3.00, 15.00), "opus": (5.00, 25.00)}
PRICING_DATE = "2026-06-24"
CHARS_PER_TOKEN = 4          # rough estimate when API usage is unavailable


def estimate_tokens(text: str | None) -> int:
    return max(1, round(len(text or "") / CHARS_PER_TOKEN)) if text else 0


def cost_usd(tier: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICING.get(tier, PRICING["haiku"])
    return in_tok / 1e6 * pin + out_tok / 1e6 * pout


def _pctl(xs: list[float], q: float) -> float:
    """Nearest-rank percentile on a small sample."""
    if not xs:
        return 0.0
    s = sorted(xs)
    import math
    k = max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))
    return s[k]


class InstrumentedLLM:
    """Wraps a base llm(prompt, system)->(text, diag) callable, recording
    per-call latency and (estimated) token volume."""

    def __init__(self, base: Callable, timer: Callable[[], float] = time.perf_counter):
        self.base = base
        self.timer = timer
        self.calls: list[dict] = []

    def __call__(self, prompt: str, system: str | None = None):
        t0 = self.timer()
        text, diag = self.base(prompt, system)
        dt = self.timer() - t0
        self.calls.append({
            "latency": dt,
            "in_tok": estimate_tokens((system or "") + prompt),
            "out_tok": estimate_tokens(text),
            "provider": diag.get("provider")})
        return text, diag


def benchmark(tier: str, cases: list[dict], base_llm: Callable,
              timer: Callable[[], float] = time.perf_counter) -> dict:
    """Run `cases` through the pipeline on one model tier. `base_llm` is a
    bare llm callable already bound to that tier."""
    per_case = []
    for c in cases:
        now = _case_now(c)
        inst = InstrumentedLLM(base_llm, timer)
        out = answer_question(c["question"], now=now, llm=inst)
        ans = out["answer"]
        latency = sum(k["latency"] for k in inst.calls)
        in_tok = sum(k["in_tok"] for k in inst.calls)
        out_tok = sum(k["out_tok"] for k in inst.calls)
        per_case.append({
            "id": c["id"],
            "latency": latency,
            "n_calls": len(inst.calls),
            "in_tok": in_tok, "out_tok": out_tok,
            "cost": cost_usd(tier, in_tok, out_tok),
            "extract_ok": ans is not None and ans.outcome == c["expect"]["outcome"],
            "mode": out["mode"]})
    return _aggregate(tier, per_case)


def _case_now(c: dict) -> datetime:
    try:
        return datetime.fromisoformat(c["now"].replace("T", " "))
    except (KeyError, ValueError, AttributeError):
        return datetime(2026, 3, 18, 12, 0)


def _aggregate(tier: str, per_case: list[dict]) -> dict:
    n = len(per_case)
    lat = [r["latency"] for r in per_case]
    polished = [r for r in per_case if r["mode"] in ("llm", "kill-switch")]
    faithful = sum(r["mode"] == "llm" for r in polished)
    mean_cost = sum(r["cost"] for r in per_case) / n if n else 0.0
    return {
        "tier": tier, "n": n, "pricing_date": PRICING_DATE,
        "token_source": "estimate(chars/4)",
        "latency_mean": round(sum(lat) / n, 4) if n else 0.0,
        "latency_p50": round(_pctl(lat, 0.50), 4),
        "latency_p95": round(_pctl(lat, 0.95), 4),
        "extraction_accuracy": round(sum(r["extract_ok"] for r in per_case) / n, 4) if n else None,
        "polish_faithful_rate": round(faithful / len(polished), 4) if polished else None,
        "polished_n": len(polished),
        "mean_calls_per_query": round(sum(r["n_calls"] for r in per_case) / n, 2) if n else 0.0,
        "mean_cost_per_query": mean_cost,
        "projected_cost_per_1k": round(mean_cost * 1000, 4),
        "per_case": per_case,
    }
