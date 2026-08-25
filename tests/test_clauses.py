import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load():
    return json.loads((ROOT / "corpus" / "parsed" / "clauses.json").read_text())


def test_store_has_all_three_regimes():
    s = load()
    regimes = {c["regime"] for c in s["clauses"]}
    assert {"2015", "jan2026-vb-sleeper", "2015-as-displayed-STALE"} <= regimes
    assert s["n_clauses"] >= 25


def test_ids_unique():
    ids = [c["id"] for c in load()["clauses"]]
    assert len(ids) == len(set(ids))


def test_calculator_critical_clauses_present():
    """The clauses the calculator will cite must exist and say what they say."""
    by_id = {c["id"]: c for c in load()["clauses"]}
    assert "cancellation" in by_id["2015/rule-6"]["text"].lower()
    assert "seventy-two hours" in by_id["jan2026/6(4)(a)"]["text"]
    assert "fifty per cent" in by_id["jan2026/6(4)(b)"]["text"]
    assert "no refund" in by_id["jan2026/6(4)(c)"]["text"]


def test_stale_exhibit_still_says_48():
    """The cached IRCTC page must contain the superseded 48-hr tier —
    if a future re-fetch updates it, this test flags the exhibit change."""
    stale = " ".join(c["text"] for c in load()["clauses"]
                     if c["regime"] == "2015-as-displayed-STALE")
    assert "48" in stale
