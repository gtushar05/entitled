"""Day-7 tests — all offline via injected fake LLMs.

Exit criteria:
1. The validator kills any polish whose numbers aren't the calculator's.
2. The deterministic template ALWAYS validates (else the kill-switch
   could kill its own fallback).
3. Extraction JSON parsing is fence/garbage-tolerant; no provider at all
   degrades to the structured-input message, never a crash.
"""

import sys
import random
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.agent import answer_case
from entitled.calculator import TicketCase
from entitled.extract import _find_json, extract_fields
from entitled.explain import polish, validate_prose
from entitled.assistant import answer_question

DEP = datetime(2026, 3, 20, 18, 0)


def mk(**kw):
    base = dict(fare=1500, cls="3A", departure=DEP,
                cancellation=DEP - timedelta(hours=24))
    base.update(kw)
    return TicketCase(**base)


def fake(reply):
    """LLM stub with the complete() signature."""
    return lambda prompt, system=None: (reply, {"provider": "fake"})


NO_LLM = lambda prompt, system=None: (None, {"provider": None})


# ------------------------------------------------------------- validator
def test_faithful_rewrite_passes():
    ans = answer_case(mk())          # refund 1125, charge 375
    good = ("Good news — you get ₹1,125 back of your ₹1,500 fare; the "
            "cancellation charge is ₹375 under 2015/rule-6.")
    ok, v = validate_prose(good, ans, mk())
    assert ok, v


def test_wrong_refund_number_is_killed():
    ans = answer_case(mk())
    bad = "You get ₹1,225 back (charge ₹375), per 2015/rule-6."
    ok, v = validate_prose(bad, ans, mk())
    assert not ok and any("1225" in x for x in v)


def test_invented_percentage_is_killed():
    ans = answer_case(mk())          # R2015: no 35% anywhere
    bad = ("You get ₹1,125 back (charge ₹375, about 35% of fare), "
           "per 2015/rule-6.")
    ok, v = validate_prose(bad, ans, mk())
    assert not ok


def test_dropped_citation_is_killed():
    ans = answer_case(mk(train_type="VBS",
                         departure=datetime(2026, 6, 10, 18, 0),
                         cancellation=datetime(2026, 6, 2, 18, 0)))
    bad = "You get ₹1,800 back of ₹2,400; the charge is ₹600 (25%)."
    ok, v = validate_prose(bad.replace("2,400", "1,500"), ans, mk())
    assert not ok  # missing jan2026/6(4)(a) citation (and wrong fare)


def test_dropped_caution_on_unverified_is_killed():
    ans = answer_case(mk(departure=datetime(2026, 6, 10, 18, 0),
                         cancellation=datetime(2026, 6, 9, 8, 0)))  # APR2026 34h
    assert ans.outcome == "COMPUTED" and not ans.verified
    confident = (f"You get ₹{int(ans.refund):,} back; charge "
                 f"₹{int(ans.charge):,}. All settled.")
    ok, v = validate_prose(confident, ans, mk())
    assert not ok and any("caution" in x for x in v)


def test_template_always_validates_property():
    """300 random cases: the kill-switch fallback must never kill itself."""
    rng = random.Random(11)
    deps = [datetime(2026, 3, 20, 18, 0), datetime(2026, 6, 10, 18, 0)]
    for _ in range(300):
        dep = rng.choice(deps)
        case = TicketCase(
            fare=rng.choice([180, 350, 750, 1500, 4000]),
            cls=rng.choice(["1A", "2A", "3A", "CC", "SL", "2S"]),
            quota=rng.choice(["GN", "GN", "TQ", "PT"]),
            status=rng.choice(["CNF", "CNF", "RAC", "WL"]),
            train_type=rng.choice(["REG", "REG", "VBS"]),
            departure=dep,
            cancellation=dep - timedelta(hours=rng.choice(
                [0.2, 2, 6, 10, 20, 30, 50, 80, 120])),
            disruption=rng.choice(["NONE"] * 8 + ["TRAIN_CANCELLED",
                                                  "DELAY_GT_3H"]),
            chart_prepared=rng.random() < 0.15)
        ans = answer_case(case)
        ok, v = validate_prose(ans.explanation, ans, case)
        assert ok, (case, v)


# ----------------------------------------------------------------- polish
def test_polish_uses_llm_when_faithful():
    ans = answer_case(mk())
    good = ("You'll get ₹1,125 of your ₹1,500 back — the charge is ₹375 "
            "(2015/rule-6).")
    text, mode, _ = polish(ans, mk(), llm=fake(good))
    assert mode == "llm" and text == good


def test_polish_kill_switch_ships_template():
    ans = answer_case(mk())
    text, mode, diag = polish(ans, mk(), llm=fake("You get ₹999 back!"))
    assert mode == "kill-switch"
    assert text == ans.explanation
    assert diag["violations"]


def test_polish_without_provider_ships_template():
    ans = answer_case(mk())
    text, mode, _ = polish(ans, mk(), llm=NO_LLM)
    assert mode == "template" and text == ans.explanation


def test_needs_info_is_never_polished():
    from entitled.agent import answer_payload
    ans = answer_payload({"cls": "3A"})
    text, mode, _ = polish(ans, None,
                           llm=fake("Just give me your PNR and relax!"))
    assert mode == "template" and text == ans.explanation


# -------------------------------------------------------------- extraction
def test_find_json_handles_fences_and_prose():
    assert _find_json('Here: ```json\n{"fare": 100}\n```')["fare"] == 100
    assert _find_json('{"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    assert _find_json("no json here") is None
    assert _find_json('[1, 2]') is None          # must be an object
    assert _find_json('{"broken": ') is None


def test_extract_fields_no_provider():
    payload, diag = extract_fields("cancel my ticket", datetime(2026, 3, 1),
                                   llm=NO_LLM)
    assert payload is None and diag["extracted"] is False


# ------------------------------------------------------------- end to end
FIELDS = ('{"fare": 1500, "cls": "3A", "departure": "2026-03-20 18:00", '
          '"cancellation": "2026-03-19 18:00"}')


def seq(*replies):
    """LLM stub returning replies in order (extract call, then polish)."""
    it = iter(replies)
    return lambda prompt, system=None: (next(it), {"provider": "fake"})


def test_answer_question_full_path_faithful():
    out = answer_question("I paid 1500 for 3A, train leaves 20 Mar 6pm, "
                          "cancelled a day before",
                          now=datetime(2026, 3, 19, 18, 0),
                          llm=seq(FIELDS,
                                  "You get ₹1,125 back; charge ₹375 "
                                  "(2015/rule-6)."))
    assert out["mode"] == "llm"
    assert out["answer"].refund == 1125
    assert out["text"].startswith("Understood: fare=1500")


def test_answer_question_lying_polisher_hits_kill_switch():
    out = answer_question("same", now=datetime(2026, 3, 19, 18, 0),
                          llm=seq(FIELDS, "Full refund of ₹1,500 for you!"))
    assert out["mode"] == "kill-switch"
    assert "₹1,125" in out["text"]        # template shipped instead


def test_answer_question_no_provider_degrades():
    out = answer_question("whatever", llm=NO_LLM)
    assert out["mode"] == "no-extractor"
    assert "provide the fields" in out["text"]


def test_answer_question_extractor_garbage_degrades():
    out = answer_question("whatever", llm=seq("I think you should call IRCTC"))
    assert out["mode"] == "no-extractor"


def test_answer_question_incomplete_extraction_asks():
    out = answer_question("cancel my sleeper ticket",
                          now=datetime(2026, 3, 1),
                          llm=seq('{"cls": "SL"}', "unused"))
    assert out["answer"].outcome == "NEEDS_INFO"
    assert "fare" in out["text"]
