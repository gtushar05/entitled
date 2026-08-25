"""Property-based sweep: invariants that must hold for EVERY case the
calculator computes, checked across hundreds of randomized inputs."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.calculator import TicketCase, compute_refund, compute_party

rng = random.Random(42)
CLASSES = ["1A", "EC", "2A", "FC", "3A", "CC", "3E", "SL", "2S"]


def random_case():
    dep_pool = [datetime(2026, 3, 20, 18, 0), datetime(2026, 6, 10, 18, 0),
                datetime(2026, 4, 8, 18, 0)]
    dep = rng.choice(dep_pool)
    train = rng.choice(["REG", "REG", "REG", "VBS", "AB2"])
    return TicketCase(
        fare=rng.choice([180, 350, 500, 750, 1000, 1500, 2400, 4000]),
        cls=rng.choice(CLASSES),
        quota=rng.choice(["GN", "GN", "GN", "TQ", "PT"]),
        status=rng.choice(["CNF", "CNF", "CNF", "RAC", "WL"]),
        channel=rng.choice(["E", "C"]),
        train_type=train,
        departure=dep,
        cancellation=dep - timedelta(hours=rng.choice(
            [0.2, 2, 6, 10, 20, 30, 50, 80, 120])),
        disruption=rng.choice(["NONE"] * 8 + ["TRAIN_CANCELLED", "DELAY_GT_3H"]),
        chart_prepared=rng.random() < 0.15,
    )


CASES = [random_case() for _ in range(600)]


def test_never_raises_and_outcomes_valid():
    for c in CASES:
        r = compute_refund(c)
        assert r.outcome in ("COMPUTED", "NO_REFUND", "ESCALATE")


def test_computed_amounts_reconcile():
    """refund + charge == fare (±1 for double rounding); nothing negative,
    nothing exceeding fare."""
    for c in CASES:
        r = compute_refund(c)
        if r.outcome == "COMPUTED":
            assert 0 <= r.refund <= c.fare + 1, (c, r)
            assert 0 <= r.charge <= c.fare + 1, (c, r)
            assert abs((r.refund + r.charge) - c.fare) <= 1, (c, r)
        if r.outcome == "NO_REFUND":
            assert r.refund == 0


def test_escalations_always_explain_themselves():
    for c in CASES:
        r = compute_refund(c)
        if r.outcome == "ESCALATE":
            assert r.notes and len(r.notes[0]) > 20


def test_earlier_cancellation_never_pays_less():
    """Within one regime, for the same ticket, cancelling earlier must never
    reduce the refund — checked on COMPUTED pairs only."""
    hours = [6, 10, 20, 30, 50, 80, 120]
    for dep, train in [(datetime(2026, 3, 20, 18, 0), "REG"),
                       (datetime(2026, 6, 10, 18, 0), "VBS")]:
        for fare in [500, 1000, 2400]:
            refunds = []
            for h in hours:
                r = compute_refund(TicketCase(
                    fare=fare, cls="3A", departure=dep, train_type=train,
                    cancellation=dep - timedelta(hours=h)))
                refunds.append(r.refund if r.outcome == "COMPUTED" else None)
            known = [x for x in refunds if x is not None]
            assert known == sorted(known), (dep, train, fare, refunds)


def test_party_sums_and_partial_tatkal_allowed():
    dep = datetime(2026, 3, 20, 18, 0)
    mk = lambda **kw: TicketCase(fare=1000, cls="3A", departure=dep,
                                 cancellation=dep - timedelta(hours=24), **kw)
    party = compute_party([mk(), mk(quota="TQ"), mk(status="WL")])
    # CNF GN: 25%=250 -> 750; CNF TQ: nil -> 0; WL: 1000-60 = 940
    assert party["total_refund"] == 750 + 0 + 940
    assert party["outcome"] == "COMPUTED"


def test_mixed_party_after_chart_escalates():
    dep = datetime(2026, 3, 20, 18, 0)
    mk = lambda **kw: TicketCase(fare=1000, cls="3A", departure=dep,
                                 cancellation=dep - timedelta(hours=5),
                                 chart_prepared=True, **kw)
    party = compute_party([mk(status="CNF"), mk(status="WL")])
    assert party["outcome"] == "ESCALATE"
