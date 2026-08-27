"""Day-9 tests — all offline (fake judge / no provider).

- cohen_kappa arithmetic on hand-worked cases.
- the validator behaves EXACTLY as documented on the ground-truth set:
  it catches wrong numbers / invented percentages / dropped citations /
  dropped cautions, and it MISSES the value-swap (the documented blind
  spot). Pinning the miss keeps the "defense in depth" story honest.
- judge_faithfulness parses a well-formed reply and returns None (never a
  silent pass) when there is no provider or the reply is garbage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.agent import answer_payload
from entitled.explain import validate_prose
from entitled.parser import parse_case
from entitled.judge import (build_faithfulness_samples, judge_faithfulness,
                            cohen_kappa)

import json
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = {c["id"]: c for c in json.loads((ROOT / "data" / "golden.json").read_text())["cases"]}


# ------------------------------------------------------------- kappa math
def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0
    # total disagreement on a balanced split -> negative
    assert cohen_kappa([True, True, False, False],
                       [False, False, True, True]) == -1.0


def test_cohen_kappa_worked_value():
    # a=b except one flip out of 10; po=0.9. a: 5T5F, b: 6T4F
    a = [True] * 5 + [False] * 5
    b = [True] * 6 + [False] * 4      # differs at index 5
    k = cohen_kappa(a, b)
    # pe = .5*.6 + .5*.4 = .5 ; kappa = (.9-.5)/(1-.5) = .8
    assert abs(k - 0.8) < 1e-9


def test_cohen_kappa_constant_labels():
    assert cohen_kappa([True, True], [True, True]) == 1.0
    assert cohen_kappa([True, True], [False, False]) == 0.0


# ------------------------------------------- validator on ground truth
def test_validator_matches_documented_behavior():
    samples = build_faithfulness_samples()
    assert len(samples) >= 15
    assert any(s["kind"] == "value_swap" for s in samples)
    for s in samples:
        c = GOLDEN[s["case_id"]]
        ans = answer_payload(c["payload"])
        case = parse_case(c["payload"]).case
        ok, violations = validate_prose(s["candidate"], ans, case)
        assert ok == s["expect_validator_pass"], (s["case_id"], s["kind"], violations)


def test_validator_catches_every_non_swap_corruption():
    for s in build_faithfulness_samples():
        if s["gt_faithful"] or s["kind"] == "value_swap":
            continue
        c = GOLDEN[s["case_id"]]
        ans = answer_payload(c["payload"])
        case = parse_case(c["payload"]).case
        ok, _ = validate_prose(s["candidate"], ans, case)
        assert not ok, f"validator missed {s['kind']} on {s['case_id']}"


def test_value_swap_is_the_blind_spot():
    """The token-set validator cannot see a refund/charge swap; this test
    documents that miss so a future 'fix' that changes it is deliberate."""
    swaps = [s for s in build_faithfulness_samples() if s["kind"] == "value_swap"]
    assert swaps
    for s in swaps:
        c = GOLDEN[s["case_id"]]
        ans = answer_payload(c["payload"])
        case = parse_case(c["payload"]).case
        ok, _ = validate_prose(s["candidate"], ans, case)
        assert ok, "value-swap unexpectedly caught — update the docs/story"


# ---------------------------------------------------------------- judge
def fake_judge(faithful: bool):
    reply = json.dumps({"faithful": faithful, "reason": "test"})
    return lambda prompt, system=None: (reply, {"provider": "fake"})


def test_judge_parses_reply():
    f, diag = judge_faithfulness("ref", "cand", llm=fake_judge(False))
    assert f is False and diag["reason"] == "test"
    f, _ = judge_faithfulness("ref", "cand", llm=fake_judge(True))
    assert f is True


def test_judge_none_without_provider_or_on_garbage():
    f, _ = judge_faithfulness("ref", "cand",
                              llm=lambda p, s=None: (None, {"provider": None}))
    assert f is None
    f, _ = judge_faithfulness("ref", "cand",
                              llm=lambda p, s=None: ("no json here", {}))
    assert f is None


def test_judge_would_catch_value_swap_with_a_perfect_judge():
    """Sanity: a judge that reads amounts flags the swap the validator misses
    (simulated with a fake that returns the ground truth)."""
    swaps = [s for s in build_faithfulness_samples() if s["kind"] == "value_swap"]
    for s in swaps:
        f, _ = judge_faithfulness(s["reference"], s["candidate"],
                                  llm=fake_judge(s["gt_faithful"]))
        assert f is False
