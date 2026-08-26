"""Agent-spine tests — Day-6 exit criteria:

1. FIDELITY: every rupee figure in the composed answer is exactly the
   calculator's output (the template can't drift from the math).
2. BINDING: clause-id citations resolve to real clause text; symbolic
   references survive unbound; nothing is dropped.
3. TOTALITY: any payload/case produces an Answer — bad intake becomes
   NEEDS_INFO, bad enum values become ESCALATE, never an exception.
"""

import sys
import random
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.agent import answer_case, answer_payload, answer_party, bind_citations
from entitled.calculator import TicketCase, compute_refund
from entitled.parser import parse_case
from entitled.retrieval import ClauseRetriever

DEP = datetime(2026, 3, 20, 18, 0)


def mk(**kw):
    base = dict(fare=1000, cls="3A", departure=DEP,
                cancellation=DEP - timedelta(hours=24))
    base.update(kw)
    return TicketCase(**base)


# ---------------------------------------------------------------- fidelity
def test_computed_answer_numbers_match_calculator_exactly():
    case = mk()
    a, r = answer_case(case), compute_refund(case)
    assert (a.refund, a.charge, a.regime, a.verified) == \
           (r.refund, r.charge, r.regime, r.verified)
    assert f"₹{r.refund:,}" in a.explanation
    assert f"₹{r.charge:,}" in a.explanation


def test_no_refund_answer_states_forfeited_fare():
    a = answer_case(mk(quota="TQ"))
    assert a.outcome == "NO_REFUND" and a.refund == 0
    assert "No refund" in a.explanation and "₹1,000" in a.explanation


def test_property_fidelity_random_sweep():
    """For 300 random cases the answer's numbers equal the calculator's,
    and COMPUTED explanations contain the exact refund figure."""
    rng = random.Random(7)
    deps = [datetime(2026, 3, 20, 18, 0), datetime(2026, 6, 10, 18, 0),
            datetime(2026, 4, 8, 18, 0)]
    for _ in range(300):
        dep = rng.choice(deps)
        case = TicketCase(
            fare=rng.choice([180, 350, 750, 1500, 4000]),
            cls=rng.choice(list("1A 2A 3A CC SL 2S".split())),
            quota=rng.choice(["GN", "GN", "TQ", "PT"]),
            status=rng.choice(["CNF", "CNF", "RAC", "WL"]),
            train_type=rng.choice(["REG", "REG", "VBS", "AB2"]),
            departure=dep,
            cancellation=dep - timedelta(hours=rng.choice(
                [0.2, 2, 6, 10, 20, 30, 50, 80, 120])),
            disruption=rng.choice(["NONE"] * 8 + ["TRAIN_CANCELLED",
                                                  "DELAY_GT_3H"]),
            chart_prepared=rng.random() < 0.15)
        a, r = answer_case(case), compute_refund(case)
        assert (a.outcome, a.refund, a.charge) == (r.outcome, r.refund, r.charge)
        assert a.explanation
        if a.outcome == "COMPUTED":
            assert f"₹{r.refund:,}" in a.explanation


# ---------------------------------------------------------------- binding
def test_vb2026_citations_bind_to_clause_text():
    a = answer_case(mk(train_type="VBS", departure=datetime(2026, 6, 10, 18, 0),
                       cancellation=datetime(2026, 6, 2, 18, 0)))  # >72h: 6(4)(a)
    bound = [c for c in a.citations if c.bound]
    assert bound and bound[0].clause_id == "jan2026/6(4)(a)"
    assert len(bound[0].text) > 50
    assert "jan2026/6(4)(a)" in a.explanation


def test_r2015_rule6_binds_despite_trailing_prose():
    a = answer_case(mk())  # cite string: "2015/rule-6 (G.S.R. 836(E)) — ..."
    assert any(c.bound and c.clause_id == "2015/rule-6" for c in a.citations)


def test_symbolic_citations_survive_unbound():
    a = answer_case(mk(quota="TQ"))  # "IRCTC Tatkal rules: ..." — no store id
    assert a.citations and all(not c.bound for c in a.citations)
    assert all(c.ref for c in a.citations)
    assert "unbound reference" in a.explanation


def test_binding_never_drops_or_reorders():
    refs = ["jan2026/6(4)(b)", "IRCTC Tatkal rules: no refund",
            "2015/rule-9 (train cancelled: full refund)"]
    out = bind_citations(refs, ClauseRetriever(use_dense=False))
    assert [c.ref for c in out] == refs
    assert [c.bound for c in out] == [True, False, True]


# ---------------------------------------------------------------- totality
def test_escalation_carries_reasons_and_needs_human():
    a = answer_case(mk(departure=datetime(2026, 6, 10, 18, 0),
                       cancellation=datetime(2026, 6, 2, 18, 0)))  # APR2026 >72h
    assert a.outcome == "ESCALATE" and a.needs_human
    assert a.refund is None
    assert any("UNVERIFIED" in n for n in a.notes)
    assert "manual review" in a.explanation


def test_invalid_enum_escalates_instead_of_crashing():
    a = answer_case(mk(cls="9Z"))
    assert a.outcome == "ESCALATE"
    assert any("9Z" in n for n in a.notes)


def test_payload_missing_fields_becomes_one_consolidated_needs_info():
    a = answer_payload({"cls": "3A"})   # no fare, no datetimes
    assert a.outcome == "NEEDS_INFO"
    assert len(a.questions) == 3        # fare + departure + cancellation
    assert "I need a few details" in a.explanation


def test_payload_aliases_and_iso_strings_round_trip():
    a = answer_payload({"fare": "1000", "class": "3AC", "quota": "Tatkal",
                        "status": "confirmed", "channel": "IRCTC",
                        "departure": "2026-03-20 18:00",
                        "cancellation": "2026-03-19T18:00"})
    assert a.outcome == "NO_REFUND"     # confirmed Tatkal


def test_date_without_time_is_blocking():
    r = parse_case({"fare": 1000, "cls": "SL",
                    "departure": "2026-03-20",
                    "cancellation": "2026-03-19 10:00"})
    assert not r.ok and any("time of day" in q for q in r.problems)


def test_unknown_value_is_named_not_guessed():
    r = parse_case({"fare": 1000, "cls": "4A",
                    "departure": "2026-03-20 18:00",
                    "cancellation": "2026-03-19 10:00"})
    assert not r.ok and any("'4A'" in q or "4A" in q for q in r.problems)


# ---------------------------------------------------------------- party
def test_party_totals_match_calculator_and_flag_escalations():
    base = {"fare": 1000, "cls": "3A", "departure": "2026-03-20 18:00",
            "cancellation": "2026-03-19 18:00"}
    out = answer_party([dict(base), dict(base, quota="TQ"),
                        dict(base, status="WL")])
    assert out["outcome"] == "COMPUTED"
    assert out["total_refund"] == 750 + 0 + 940
    assert "₹1,690" in out["explanation"]
    assert len(out["answers"]) == 3


def test_party_one_bad_passenger_blocks_with_named_index():
    ok = {"fare": 1000, "cls": "3A", "departure": "2026-03-20 18:00",
          "cancellation": "2026-03-19 18:00"}
    out = answer_party([ok, {"cls": "3A"}])
    assert out["outcome"] == "NEEDS_INFO"
    assert any(q.startswith("passenger 2:") for q in out["questions"])


# ------------------------------------------- review-panel regressions (Day 6)
def test_timezone_aware_datetimes_are_normalized_not_crashing():
    # panel finding: aware/naive mix raised TypeError (totality violation)
    a = answer_payload({"fare": 1000, "cls": "3A",
                        "departure": "2026-03-20 18:00",
                        "cancellation": "2026-03-19T18:00:00+05:30"})
    assert a.outcome == "COMPUTED" and a.refund == 750
    b = answer_payload({"fare": 1000, "cls": "3A",
                        "departure": "2026-03-20T18:00:00Z",   # UTC
                        "cancellation": "2026-03-19 18:00"})
    assert b.outcome in ("COMPUTED", "NO_REFUND")
    assert any("non-IST" in n for n in b.notes)   # conversion is surfaced


def test_plain_amrit_bharat_is_ambiguous_not_guessed():
    # panel finding: 'AMRIT BHARAT' -> AB2 misapplied the Jan-2026 regime
    # to original Amrit Bharat trains, which run under regular rules
    a = answer_payload({"fare": 1000, "cls": "SL", "train_type": "Amrit Bharat",
                        "departure": "2026-03-20 18:00",
                        "cancellation": "2026-03-19 18:00"})
    assert a.outcome == "NEEDS_INFO"
    assert any("2.0" in q for q in a.questions)


def test_unreserved_is_out_of_scope_not_mapped_to_2s():
    a = answer_payload({"fare": 100, "cls": "unreserved",
                        "departure": "2026-03-20 18:00",
                        "cancellation": "2026-03-19 18:00"})
    assert a.outcome == "NEEDS_INFO"
    assert any("unreserved" in q.lower() for q in a.questions)


def test_nonfinite_fare_blocked_at_intake():
    for fare in ["inf", "Infinity", float("inf"), float("nan"), "9" * 400]:
        a = answer_payload({"fare": fare, "cls": "3A",
                            "departure": "2026-03-20 18:00",
                            "cancellation": "2026-03-19 18:00"})
        assert a.outcome == "NEEDS_INFO", fare


def test_hostile_payloads_never_raise():
    for payload in [None, 42, "cancel my ticket", [{"fare": 1}],
                    {1: "x", "fare": 1000}, {None: "x"}, {}]:
        a = answer_payload(payload)
        assert a.outcome == "NEEDS_INFO"
    for party in [None, [], "x", [None], [{"fare": 1000}]]:
        out = answer_party(party)
        assert out["outcome"] == "NEEDS_INFO"


def test_answer_case_total_beyond_valueerror():
    # panel finding: TypeError/OverflowError leaked through the ValueError-only
    # handler; fare=None reaches the calculator's arithmetic
    a = answer_case(mk(fare=None))
    assert a.outcome == "ESCALATE"


def test_money_rounds_half_up_like_calculator():
    from entitled.agent import _money
    assert _money(1000.5) == "₹1,001"      # round() would give ₹1,000
    assert _money(0.5) == "₹1"


def test_party_dict_shape_is_uniform_across_branches():
    keys = {"outcome", "answers", "total_refund", "escalated_passengers",
            "questions", "notes", "explanation"}
    ok = {"fare": 1000, "cls": "3A", "departure": "2026-03-20 18:00",
          "cancellation": "2026-03-19 18:00"}
    chart = dict(ok, cancellation="2026-03-20 15:00", chart_prepared=True)
    for out in [answer_party([ok]),                       # COMPUTED
                answer_party([{"cls": "3A"}]),            # NEEDS_INFO
                answer_party([dict(chart, status="CNF"),  # whole-party ESCALATE
                              dict(chart, status="WL")]),
                answer_party([ok, dict(ok, departure="2026-06-10 18:00",
                                       cancellation="2026-06-05 18:00")])]:
        assert set(out) == keys, out["outcome"]


def test_party_escalated_header_and_warnings_survive():
    ok = {"fare": 1000, "cls": "3A", "departure": "2026-03-20 18:00",
          "cancellation": "2026-03-19 18:00", "pnr_color": "blue"}
    apr = {"fare": 1000, "cls": "3A", "departure": "2026-06-10 18:00",
           "cancellation": "2026-06-05 18:00"}   # APR2026 >72h: ESCALATE
    out = answer_party([ok, apr])
    assert out["outcome"] == "ESCALATE"
    assert out["escalated_passengers"] == [1]
    assert "need manual review" in out["explanation"]
    # per-passenger parse warnings must reach the party answers (finding #5)
    assert any("pnr_color" in n for n in out["answers"][0].notes)


def test_explicit_null_cls_falls_back_to_class_field():
    a = answer_payload({"fare": 1000, "cls": None, "class": "3A",
                        "departure": "2026-03-20 18:00",
                        "cancellation": "2026-03-19 18:00"})
    assert a.outcome == "COMPUTED"


def test_party_mixed_status_post_chart_escalates_whole():
    base = {"fare": 1000, "cls": "3A", "departure": "2026-03-20 18:00",
            "cancellation": "2026-03-20 15:00", "chart_prepared": True}
    out = answer_party([dict(base, status="CNF"), dict(base, status="WL")])
    assert out["outcome"] == "ESCALATE"
    assert "manual review" in out["explanation"]
