"""Calculator tests — every expected value hand-computed from the rules.

Convention: D = departure 2026-XX-XX; cancellation set by hours-before.
Pre-Apr-2026 cases use March 2026 dates (R2015 regime for regular trains).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.calculator import TicketCase, compute_refund

D_2015 = datetime(2026, 3, 20, 18, 0)     # regular-train departure, R2015 era
D_APR = datetime(2026, 6, 10, 18, 0)      # post-rollout departure, APR2026 era
D_VB = datetime(2026, 6, 10, 18, 0)


def case(fare, cls="3A", h_before=60, dep=D_2015, **kw):
    return TicketCase(fare=fare, cls=cls, departure=dep,
                      cancellation=dep - timedelta(hours=h_before), **kw)


# ---------------- regime selection ----------------
def test_regular_pre_april_is_2015():
    assert compute_refund(case(1000, h_before=60)).regime == "R2015"

def test_regular_post_april_is_apr2026():
    assert compute_refund(case(1000, h_before=30, dep=D_APR)).regime == "APR2026"

def test_rollout_window_escalates():
    dep = datetime(2026, 4, 10, 18, 0)
    r = compute_refund(case(1000, h_before=30, dep=dep))
    assert r.outcome == "ESCALATE" and "rollout" in r.notes[0]

def test_vb_sleeper_uses_jan2026_regime():
    r = compute_refund(case(2000, h_before=100, dep=D_VB, train_type="VBS"))
    assert r.regime == "VB2026"


# ---------------- R2015 confirmed tiers (hand-computed) ----------------
def test_2015_early_flat_only_sl():
    # SL fare 1000, 60h before: flat Rs 120 -> refund 880
    r = compute_refund(case(1000, cls="SL", h_before=60))
    assert (r.refund, r.charge, r.verified) == (880, 120, True)

def test_2015_25pct_binds_over_flat():
    # 3A fare 1000, 24h before: 25% = 250 > 180 -> refund 750
    r = compute_refund(case(1000, cls="3A", h_before=24))
    assert (r.refund, r.charge) == (750, 250)

def test_2015_flat_minimum_binds_small_fare():
    # 3A fare 500, 24h before: 25% = 125 < 180 -> charge 180 -> refund 320
    r = compute_refund(case(500, cls="3A", h_before=24))
    assert (r.refund, r.charge) == (320, 180)

def test_2015_50pct_tier():
    # 2S fare 300, 6h before: 50% = 150 > 60 -> refund 150
    r = compute_refund(case(300, cls="2S", h_before=6))
    assert (r.refund, r.charge) == (150, 150)

def test_2015_under_4h_no_refund():
    r = compute_refund(case(1000, h_before=3))
    assert r.outcome == "NO_REFUND" and r.refund == 0


# ---------------- VB2026 (VERIFIED from CC 08/2026) ----------------
def test_vb_early_costs_25pct():
    r = compute_refund(case(2000, h_before=100, dep=D_VB, train_type="VBS"))
    assert (r.refund, r.charge) == (1500, 500)
    assert r.citations == ["jan2026/6(4)(a)"] and r.verified

def test_vb_mid_costs_50pct():
    r = compute_refund(case(2000, h_before=24, dep=D_VB, train_type="VBS"))
    assert (r.refund, r.charge) == (1000, 1000)
    assert r.citations == ["jan2026/6(4)(b)"]

def test_vb_under_8h_nil():
    r = compute_refund(case(2000, h_before=5, dep=D_VB, train_type="VBS"))
    assert r.outcome == "NO_REFUND" and r.citations == ["jan2026/6(4)(c)"]

def test_amrit_bharat_2_same_regime():
    r = compute_refund(case(1000, h_before=100, dep=D_VB, train_type="AB2"))
    assert r.regime == "VB2026" and r.charge == 250


# ---------------- APR2026: honest verification boundaries ----------------
def test_apr_25pct_safe_when_flat_nonbinding():
    # 3A fare 2000, 30h: 25% = 500 > claimed flat 180 -> COMPUTED (not verified)
    r = compute_refund(case(2000, cls="3A", h_before=30, dep=D_APR))
    assert (r.outcome, r.refund, r.verified) == ("COMPUTED", 1500, False)

def test_apr_small_fare_escalates_on_unverified_flat():
    # 3A fare 500, 30h: 25% = 125 <= 180 claimed -> the flat decides -> ESCALATE
    r = compute_refund(case(500, cls="3A", h_before=30, dep=D_APR))
    assert r.outcome == "ESCALATE" and "UNVERIFIED" in r.notes[0]

def test_apr_flat_only_tier_escalates():
    r = compute_refund(case(2000, h_before=100, dep=D_APR))
    assert r.outcome == "ESCALATE"

def test_apr_under_8h_nil():
    r = compute_refund(case(2000, h_before=6, dep=D_APR))
    assert r.outcome == "NO_REFUND"


# ---------------- RAC / WL ----------------
def test_rac_clerkage_before_cutoff():
    r = compute_refund(case(800, status="RAC", h_before=2))
    assert (r.refund, r.charge) == (740, 60)

def test_wl_past_30min_cutoff_nil():
    r = compute_refund(case(800, status="WL", h_before=0.3))  # 18 min
    assert r.outcome == "NO_REFUND"

def test_wl_eticket_autocancel_note():
    r = compute_refund(case(800, status="WL", h_before=5, channel="E"))
    assert any("auto-cancelled" in n for n in r.notes)


# ---------------- quotas ----------------
def test_tatkal_cnf_no_refund():
    r = compute_refund(case(900, quota="TQ", h_before=60))
    assert r.outcome == "NO_REFUND"

def test_tatkal_waitlist_gets_clerkage_path():
    r = compute_refund(case(900, quota="TQ", status="WL", h_before=5))
    assert (r.outcome, r.refund) == ("COMPUTED", 840)

def test_premium_tatkal_cnf_no_refund():
    r = compute_refund(case(1500, quota="PT", h_before=100))
    assert r.outcome == "NO_REFUND"


# ---------------- disruption overrides ----------------
def test_train_cancelled_full_refund_any_quota():
    r = compute_refund(case(900, quota="TQ", h_before=1,
                            disruption="TRAIN_CANCELLED"))
    assert (r.refund, r.charge, r.verified) == (900, 0, True)

def test_delay_gt3h_full_refund_with_actual_departure_note():
    r = compute_refund(case(1200, h_before=1, disruption="DELAY_GT_3H"))
    assert r.refund == 1200
    assert any("ACTUAL" in n for n in r.notes)

def test_delay_but_travelled_escalates():
    r = compute_refund(case(1200, h_before=1, disruption="DELAY_GT_3H",
                            travelled=True))
    assert r.outcome == "ESCALATE"


# ---------------- edges ----------------
def test_vb_before_regime_start_escalates():
    dep = datetime(2026, 1, 20, 10, 0)
    c = TicketCase(fare=1000, cls="3A", departure=dep, train_type="VBS",
                   cancellation=datetime(2026, 1, 10, 10, 0))
    assert compute_refund(c).outcome == "ESCALATE"

def test_invalid_class_raises():
    with pytest.raises(ValueError):
        compute_refund(case(1000, cls="XX"))
