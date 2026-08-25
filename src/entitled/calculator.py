"""The deterministic, version-aware refund calculator — Entitled's core.

Design invariants:
- The LLM NEVER does arithmetic. Every rupee figure in any answer comes
  from this module, or the answer doesn't ship.
- Every branch cites its source (clause ids where verified in the clause
  store, symbolic references otherwise — bound to store ids by retrieval).
- The calculator knows what it doesn't know: rule regimes with unverified
  primary text, ambiguous rollout windows, and out-of-scope cases return
  ESCALATE with the closest governing citation — never a guess.

Regimes (see corpus/MANIFEST.json verification_log):
  R2015   — G.S.R. 836(E): VERIFIED (gazette OCR'd in corpus).
  VB2026  — G.S.R. 41(E) via CC 08/2026: VERIFIED from primary text layer.
  APR2026 — announced reform (72/24/8, all trains, phased 1–15 Apr 2026):
            percentage tiers confirmed by the government's own announcement;
            PRIMARY circular not yet published on the CC index → any branch
            whose outcome depends on the (unconfirmed) flat minimum charges
            ESCALATES per the pre-committed rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

# ------------------------------------------------------------------ constants
# Flat cancellation charges per passenger, 2015 rules (G.S.R. 836(E), rule 6):
# also the *claimed* (unverified) flats for the Apr-2026 regime.
FLAT_2015 = {"1A": 240, "EC": 240, "2A": 200, "FC": 200,
             "3A": 180, "CC": 180, "3E": 180, "SL": 120, "2S": 60}
CLERKAGE_RESERVED = 60      # RAC/WL, per passenger (2015 rules; unchanged by reforms)

APR2026_ROLLOUT_START = datetime(2026, 4, 1)
APR2026_ROLLOUT_END = datetime(2026, 4, 15, 23, 59)
VB2026_EFFECTIVE = datetime(2026, 1, 16)

VALID_CLASSES = set(FLAT_2015)
VALID_QUOTAS = {"GN", "TQ", "PT"}          # general, tatkal, premium tatkal
VALID_STATUS = {"CNF", "RAC", "WL"}
VALID_TRAIN = {"REG", "VBS", "AB2"}         # regular, VB Sleeper, Amrit Bharat II
VALID_DISRUPTION = {"NONE", "TRAIN_CANCELLED", "DELAY_GT_3H"}


@dataclass
class TicketCase:
    fare: float                      # per passenger, base fare paid
    cls: str                         # travel class code
    quota: str = "GN"
    status: str = "CNF"
    channel: str = "E"               # E = e-ticket, C = counter
    train_type: str = "REG"
    departure: datetime = None       # scheduled departure
    cancellation: datetime = None    # when cancellation/TDR is presented
    disruption: str = "NONE"
    travelled: bool = False
    chart_prepared: bool = False     # first chart is now ≥10h pre-departure (Dec-2025)


@dataclass
class Result:
    outcome: str                     # COMPUTED | NO_REFUND | ESCALATE
    refund: float | None
    charge: float | None
    regime: str
    verified: bool                   # primary-source-verified branch?
    citations: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _r(x: float) -> int:
    """Round to whole rupees, half up (documented convention)."""
    return int(x + 0.5)


def _regime(case: TicketCase) -> tuple[str | None, list[str]]:
    if case.train_type in ("VBS", "AB2"):
        if case.cancellation >= VB2026_EFFECTIVE:
            return "VB2026", []
        return None, ["cancellation predates the VB-Sleeper/AB-II regime "
                      "(G.S.R. 41(E), w.e.f. 16.01.2026) — out of scope"]
    if case.cancellation < APR2026_ROLLOUT_START:
        return "R2015", []
    if case.cancellation <= APR2026_ROLLOUT_END:
        return None, ["cancellation falls inside the phased Apr-2026 rollout "
                      "window (1–15 Apr) — which regime applied depends on "
                      "train/zone rollout date; cannot be determined from rules alone"]
    return "APR2026", []


def compute_refund(case: TicketCase) -> Result:
    # ---- validation ----
    for val, ok, what in [(case.cls, VALID_CLASSES, "class"),
                          (case.quota, VALID_QUOTAS, "quota"),
                          (case.status, VALID_STATUS, "status"),
                          (case.train_type, VALID_TRAIN, "train_type"),
                          (case.disruption, VALID_DISRUPTION, "disruption")]:
        if val not in ok:
            raise ValueError(f"invalid {what}: {val}")
    if case.departure is None or case.cancellation is None:
        raise ValueError("departure and cancellation datetimes are required")

    regime, why = _regime(case)
    if regime is None:
        return Result("ESCALATE", None, None, "UNDETERMINED", False, notes=why)

    hours_before = (case.departure - case.cancellation).total_seconds() / 3600

    # ---- disruption overrides (any regime) ----
    if case.disruption == "TRAIN_CANCELLED":
        note = ("e-ticket: full refund is automatic, no action needed"
                if case.channel == "E" else
                "counter ticket: surrender at any counter within 72h of scheduled departure")
        return Result("COMPUTED", _r(case.fare), 0, regime, True,
                      citations=["2015/rule-9 (train cancelled: full refund)"],
                      notes=[note])
    if case.disruption == "DELAY_GT_3H":
        if case.travelled:
            return Result("ESCALATE", None, None, regime, False,
                          notes=["delay >3h but passenger travelled — no refund "
                                 "under the late-running rule; partial-journey "
                                 "claims are out of calculator scope"])
        note = ("e-ticket: file TDR online BEFORE the ACTUAL departure of the train"
                if case.channel == "E" else
                "counter ticket: surrender up to the ACTUAL departure of the train")
        return Result("COMPUTED", _r(case.fare), 0, regime, True,
                      citations=["2015 Rules — late running >3 hours: full refund"],
                      notes=[note, "note: the cutoff is the ACTUAL departure, "
                                   "not the scheduled one — commonly misreported"])

    # ---- RAC / WL (regime-independent; reforms left these unchanged) ----
    if case.status in ("RAC", "WL"):
        if case.quota == "PT":
            return Result("ESCALATE", None, None, regime, False,
                          notes=["Premium Tatkal issues only CNF/RAC; RAC/WL PT "
                                 "combinations need manual review"]) \
                if case.status == "WL" else _rac_wl(case, regime, hours_before)
        return _rac_wl(case, regime, hours_before)

    # ---- chart-preparation scope boundary (CNF only; RAC/WL handled above) ----
    # After chart, online cancellation is blocked and refunds route via TDR
    # with case-specific rules (and chart now happens ≥10h before departure,
    # i.e. earlier than several tier boundaries). Post-chart CNF cancellation
    # is outside the calculator's verified scope — escalate, don't guess.
    if case.chart_prepared:
        return Result("ESCALATE", None, None, regime, False,
                      notes=["chart already prepared: CNF cancellation refunds "
                             "route via TDR with case-specific rules — outside "
                             "the calculator's verified scope",
                             "note: since Dec-2025, first chart is prepared at "
                             "least 10 hours before departure"])

    # ---- CNF quota gates ----
    if case.quota == "TQ":
        return Result("NO_REFUND", 0, _r(case.fare), regime, True,
                      citations=["IRCTC Tatkal rules: no refund on cancellation "
                                 "of confirmed Tatkal tickets"],
                      notes=["exceptions (train cancelled / >3h late) are handled "
                             "by the disruption branches"])
    if case.quota == "PT":
        return Result("NO_REFUND", 0, _r(case.fare), regime, True,
                      citations=["Premium Tatkal: no refund on confirmed tickets"],
                      notes=["dynamic fare fully forfeited"])

    # ---- CNF general-quota tiers, by regime ----
    if regime == "R2015":
        return _cnf_2015(case, hours_before)
    if regime == "VB2026":
        return _cnf_vb2026(case, hours_before)
    return _cnf_apr2026(case, hours_before)


def _rac_wl(case: TicketCase, regime: str, hours_before: float) -> Result:
    cite = ["2015 Rules — RAC/WL: refund of fare less clerkage (Rs 60/passenger), "
            "up to 30 minutes before scheduled departure"]
    if hours_before * 60 < 30:
        note = ("past the 30-minute cutoff — no refund on cancellation; "
                "e-ticket TDR paths for exceptional cases need manual review"
                if case.channel == "E" else
                "past the 30-minute cutoff — no refund")
        return Result("NO_REFUND", 0, _r(case.fare), regime, True,
                      citations=cite, notes=[note])
    notes = []
    if case.status == "WL" and case.channel == "E":
        notes.append("fully-WL e-tickets are auto-cancelled at chart preparation "
                     "with the same clerkage — no user action needed")
    if case.status == "RAC" and case.channel == "E":
        notes.append("RAC e-tickets are NOT auto-cancelled — cancel or file TDR "
                     "before the 30-minute cutoff or forfeit")
    return Result("COMPUTED", _r(case.fare - CLERKAGE_RESERVED),
                  CLERKAGE_RESERVED, regime, True, citations=cite, notes=notes)


def _cnf_2015(case: TicketCase, h: float) -> Result:
    flat = FLAT_2015[case.cls]
    cite = ["2015/rule-6 (G.S.R. 836(E)) — confirmed-ticket cancellation charges"]
    if h > 48:
        charge = flat
    elif h > 12:
        charge = max(0.25 * case.fare, flat)
    elif h > 4:
        charge = max(0.50 * case.fare, flat)
    else:
        return Result("NO_REFUND", 0, _r(case.fare), "R2015", True, citations=cite,
                      notes=["less than 4 hours before scheduled departure"])
    if charge >= case.fare:
        # flat minimum equals/exceeds the fare — the charge is capped at the
        # fare (a cancellation can never cost more than was paid)
        return Result("NO_REFUND", 0, _r(case.fare), "R2015", True, citations=cite,
                      notes=["flat cancellation charge equals or exceeds the fare"])
    notes = ["GST on AC-class cancellation charges applies extra (not computed)"] \
        if case.cls in ("1A", "EC", "2A", "FC", "3A", "CC", "3E") else []
    return Result("COMPUTED", _r(case.fare - charge), _r(charge), "R2015", True,
                  citations=cite, notes=notes)


def _cnf_vb2026(case: TicketCase, h: float) -> Result:
    if h > 72:
        charge, cite = 0.25 * case.fare, ["jan2026/6(4)(a)"]
    elif h > 8:
        charge, cite = 0.50 * case.fare, ["jan2026/6(4)(b)"]
    else:
        return Result("NO_REFUND", 0, _r(case.fare), "VB2026", True,
                      citations=["jan2026/6(4)(c)"],
                      notes=["less than 8 hours before scheduled departure"])
    return Result("COMPUTED", _r(case.fare - charge), _r(charge), "VB2026", True,
                  citations=cite,
                  notes=["note: no flat-charge-only band — even early cancellation "
                         "costs 25% on these trains (rule 6(4))"])


def _cnf_apr2026(case: TicketCase, h: float) -> Result:
    """Apr-2026 all-trains regime. Percentage tiers: confirmed by the
    government's announcement (primary circular pending). Flat minimums:
    UNVERIFIED → any branch they decide ESCALATES (pre-committed rule)."""
    claimed_flat = FLAT_2015[case.cls]
    cite = ["Apr-2026 reform (72/24/8) — NewsOnAir gov announcement; "
            "primary circular pending on CC index"]
    if h > 72:
        return Result("ESCALATE", None, None, "APR2026", False, citations=cite,
                      notes=[">72h tier is flat-charge-only; the flat charges for "
                             "this regime are UNVERIFIED in primary sources — "
                             "escalating rather than guessing (pre-committed rule)"])
    if h > 24:
        pct = 0.25
    elif h > 8:
        pct = 0.50
    else:
        return Result("NO_REFUND", 0, _r(case.fare), "APR2026", False, citations=cite,
                      notes=["less than 8 hours before scheduled departure "
                             "(tier confirmed by gov announcement; primary pending)"])
    pct_charge = pct * case.fare
    if pct_charge <= claimed_flat:
        return Result("ESCALATE", None, None, "APR2026", False, citations=cite,
                      notes=[f"{int(pct*100)}% of fare (Rs {_r(pct_charge)}) does not "
                             f"exceed the claimed flat minimum (Rs {claimed_flat}); "
                             "the binding charge depends on UNVERIFIED flat minimums "
                             "— escalating (pre-committed rule)"])
    return Result("COMPUTED", _r(case.fare - pct_charge), _r(pct_charge),
                  "APR2026", False, citations=cite,
                  notes=["flat minimum non-binding here under any claimed value "
                         "(percentage charge exceeds it), so the amount is safe "
                         "despite the pending primary"])


def compute_party(cases: list[TicketCase]) -> dict:
    """Party/partial cancellation: per-passenger computation.

    Pre-chart, partial cancellation of e-tickets (Tatkal included) is allowed
    passenger-by-passenger at the applicable slab — so the party result is the
    sum of individual results. The special mixed-status post-chart rule
    (confirmed passengers on a party ticket where others are WL: full refund
    less clerkage, 30-min cutoff, all passengers on one TDR) is out of the
    verified scope and escalates as a party.
    """
    if any(c.chart_prepared for c in cases):
        statuses = {c.status for c in cases}
        if len(statuses) > 1:
            return {"outcome": "ESCALATE", "results": [],
                    "notes": ["mixed-status party ticket after chart: the "
                              "special full-refund-less-clerkage rule applies "
                              "with a single TDR for all passengers — manual "
                              "review required"]}
    results = [compute_refund(c) for c in cases]
    total_refund = sum(r.refund for r in results if r.refund) or 0
    escalations = [i for i, r in enumerate(results) if r.outcome == "ESCALATE"]
    return {
        "outcome": "ESCALATE" if escalations else "COMPUTED",
        "results": results,
        "total_refund": int(total_refund),
        "escalated_passengers": escalations,
        "notes": ["fresh e-reservation slip must be printed after partial "
                  "cancellation"] if len(cases) > 1 else [],
    }
