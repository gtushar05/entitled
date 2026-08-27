"""Day-10 benchmark-harness tests — all offline (fake provider + fake
timer, so latency and quality are deterministic)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.bench import (estimate_tokens, cost_usd, _pctl, PRICING,
                            InstrumentedLLM, benchmark)

import json
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = {c["id"]: c for c in json.loads((ROOT / "data" / "golden.json").read_text())["cases"]}


# ---------------------------------------------------------------- cost model
def test_cost_usd_matches_published_pricing():
    # 1M input + 1M output on haiku = $1 + $5
    assert abs(cost_usd("haiku", 1_000_000, 1_000_000) - 6.00) < 1e-9
    assert abs(cost_usd("sonnet", 1_000_000, 0) - 3.00) < 1e-9
    assert abs(cost_usd("opus", 0, 1_000_000) - 25.00) < 1e-9


def test_pricing_tiers_ordered():
    assert PRICING["haiku"] < PRICING["sonnet"] < PRICING["opus"]


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("a" * 40) == 10


def test_percentile_nearest_rank():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _pctl(xs, 0.50) == 5
    assert _pctl(xs, 0.95) == 10
    assert _pctl([], 0.5) == 0.0


# ----------------------------------------------------- instrumented llm
def test_instrumented_llm_records_calls():
    ticks = iter([0.0, 0.5, 1.0, 3.0])          # start,end pairs
    base = lambda p, s=None: ("reply text here", {"provider": "fake"})
    inst = InstrumentedLLM(base, timer=lambda: next(ticks))
    inst("hello", "sys")
    inst("world")
    assert [round(c["latency"], 3) for c in inst.calls] == [0.5, 2.0]
    assert inst.calls[0]["provider"] == "fake"
    assert inst.calls[0]["out_tok"] == estimate_tokens("reply text here")


# ------------------------------------------------------------- benchmark
FIELDS = ('{"fare": 1500, "cls": "3A", "departure": "2026-03-20 18:00", '
          '"cancellation": "2026-03-19 18:00"}')


def make_llm(extract_reply, polish_reply, dt=1.0):
    """Deterministic fake: 1st call (extract) returns fields, 2nd (polish)
    returns prose; each call advances the fake clock by `dt`."""
    state = {"n": 0}
    def base(prompt, system=None):
        state["n"] += 1
        return (extract_reply if state["n"] % 2 == 1 else polish_reply,
                {"provider": "fake"})
    return base


def fake_timer(step=1.0):
    t = {"v": 0.0}
    def clock():
        v = t["v"]; t["v"] += step
        return v
    return clock


def test_benchmark_aggregates_quality_and_cost():
    cases = [GOLDEN["r2015-3a-36h"]]          # expect COMPUTED, refund 1125
    faithful_polish = ("You get ₹1,125 back of ₹1,500 "
                       "(charge ₹375, 2015/rule-6).")
    base = make_llm(FIELDS, faithful_polish)
    r = benchmark("haiku", cases, base, timer=fake_timer(1.0))
    assert r["tier"] == "haiku" and r["n"] == 1
    assert r["extraction_accuracy"] == 1.0
    assert r["polish_faithful_rate"] == 1.0        # faithful prose -> mode llm
    assert r["mean_calls_per_query"] == 2.0        # extract + polish
    assert r["mean_cost_per_query"] > 0
    assert r["latency_mean"] > 0


def test_benchmark_counts_killed_polish_as_unfaithful():
    cases = [GOLDEN["r2015-3a-36h"]]
    base = make_llm(FIELDS, "You get a full refund of ₹9,999!")  # lie
    r = benchmark("haiku", cases, base, timer=fake_timer())
    assert r["extraction_accuracy"] == 1.0         # extraction still fine
    assert r["polish_faithful_rate"] == 0.0        # polish killed
    assert r["per_case"][0]["mode"] == "kill-switch"


def test_benchmark_cost_scales_with_tier():
    cases = [GOLDEN["r2015-3a-36h"]]
    base_h = make_llm(FIELDS, "₹1,125 back; charge ₹375 (2015/rule-6).")
    base_o = make_llm(FIELDS, "₹1,125 back; charge ₹375 (2015/rule-6).")
    rh = benchmark("haiku", cases, base_h, timer=fake_timer())
    ro = benchmark("opus", cases, base_o, timer=fake_timer())
    assert ro["mean_cost_per_query"] > rh["mean_cost_per_query"]
