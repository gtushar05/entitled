"""Structured intake -> validated TicketCase.

This is the agent's trust boundary. Whatever produces the payload — the
LLM field-extractor (Day 7), a web form, a test — nothing reaches the
calculator without passing through here. The LLM proposes fields; this
module disposes. Unknown values are never silently 'fixed': they come
back as field-level problems the agent turns into a clarifying question
(NEEDS_INFO), because a guessed class or a guessed hour changes rupees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

from .calculator import (TicketCase, VALID_CLASSES, VALID_QUOTAS,
                         VALID_STATUS, VALID_TRAIN, VALID_DISRUPTION)

# ---- alias tables: common human/LLM phrasings -> canonical codes ----
CLASS_ALIASES = {
    "1AC": "1A", "FIRST AC": "1A", "AC FIRST": "1A", "AC 1 TIER": "1A",
    "EXECUTIVE": "EC", "EXECUTIVE CHAIR CAR": "EC", "EXEC": "EC",
    "2AC": "2A", "SECOND AC": "2A", "AC 2 TIER": "2A", "AC TWO TIER": "2A",
    "FIRST CLASS": "FC",
    "3AC": "3A", "THIRD AC": "3A", "AC 3 TIER": "3A", "AC THREE TIER": "3A",
    "CHAIR CAR": "CC", "AC CHAIR CAR": "CC",
    "AC 3 ECONOMY": "3E", "3AC ECONOMY": "3E", "ECONOMY": "3E",
    "SLEEPER": "SL", "SLEEPER CLASS": "SL",
    "SECOND SITTING": "2S", "SECOND SEATING": "2S",
}
# values that LOOK mappable but must be asked about, not guessed — mapping
# them silently would change which rules (or whether these rules) apply
AMBIGUOUS = {
    "UNRESERVED": "class: unreserved tickets are outside the reserved-ticket "
                  "cancellation rules this calculator covers — is this a "
                  "reserved Second Sitting (2S) ticket?",
    "AMRIT BHARAT": "train_type: ambiguous — Amrit Bharat 2.0 falls under the "
                    "Jan-2026 rules (AB2) but the original Amrit Bharat under "
                    "regular rules (REG); which is it?",
}
QUOTA_ALIASES = {"GENERAL": "GN", "TATKAL": "TQ", "PREMIUM TATKAL": "PT"}
STATUS_ALIASES = {"CONFIRMED": "CNF", "WAITLIST": "WL", "WAITLISTED": "WL",
                  "WAITING LIST": "WL", "RAC": "RAC"}
TRAIN_ALIASES = {"REGULAR": "REG", "MAIL": "REG", "EXPRESS": "REG",
                 "VANDE BHARAT SLEEPER": "VBS", "VB SLEEPER": "VBS",
                 "AMRIT BHARAT 2.0": "AB2", "AMRIT BHARAT II": "AB2"}
CHANNEL_ALIASES = {"ONLINE": "E", "IRCTC": "E", "E-TICKET": "E", "ETICKET": "E",
                   "COUNTER": "C", "PRS": "C", "WINDOW": "C", "STATION": "C"}
DISRUPTION_ALIASES = {"": "NONE", "CANCELLED BY RAILWAYS": "TRAIN_CANCELLED",
                      "TRAIN CANCELLED": "TRAIN_CANCELLED",
                      "LATE MORE THAN 3 HOURS": "DELAY_GT_3H",
                      "DELAYED OVER 3 HOURS": "DELAY_GT_3H"}


@dataclass
class ParseResult:
    case: TicketCase | None            # None when problems is non-empty
    problems: list[str] = field(default_factory=list)   # blocking; ask the user
    warnings: list[str] = field(default_factory=list)   # non-blocking caveats

    @property
    def ok(self) -> bool:
        return self.case is not None


def _canon(value, aliases: dict, valid: set) -> str | None:
    """Uppercase, try the alias table, then the canonical set. None = unknown."""
    v = str(value).strip().upper()
    v = aliases.get(v, v)
    return v if v in valid else None


def _dt(value, name: str, problems: list[str], warnings: list[str]):
    if isinstance(value, datetime):
        return _naive_ist(value, name, warnings)
    if value is None or str(value).strip() == "":
        problems.append(f"{name}: missing — required to place the case on the "
                        "cancellation-charge tiers")
        return None
    s = str(value).strip().replace("T", " ")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        problems.append(f"{name}: could not parse '{value}' — expected "
                        "YYYY-MM-DD HH:MM")
        return None
    if len(s) <= 10:  # date only, no clock time
        problems.append(f"{name}: '{value}' has no time of day — the charge "
                        "tiers are hour-sensitive (48/12/4h etc.), so the "
                        "clock time is required")
        return None
    return _naive_ist(dt, name, warnings)


def _naive_ist(dt: datetime, name: str, warnings: list[str]) -> datetime:
    """Normalize timezone-aware datetimes to naive IST — all rule tiers are
    in Indian railway time, and mixing aware/naive values crashes datetime
    arithmetic (a totality violation the review panel caught)."""
    if dt.tzinfo is None:
        return dt
    if dt.utcoffset() != IST.utcoffset(None):
        warnings.append(f"{name}: converted from a non-IST timezone to IST — "
                        "verify the intended local time")
    return dt.astimezone(IST).replace(tzinfo=None)


def parse_case(payload: dict) -> ParseResult:
    """Validate one passenger's payload into a TicketCase.

    Blocking problems (missing fare, unparseable datetime, unknown class)
    return case=None with every problem listed at once — the agent asks a
    single consolidated clarifying question, not one per turn.
    """
    problems: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return ParseResult(None, [f"payload must be a JSON object with the "
                                  f"ticket fields, got {type(payload).__name__}"])
    p = {str(k).strip().lower(): v for k, v in payload.items()}

    known = {"fare", "cls", "class", "quota", "status", "channel",
             "train_type", "departure", "cancellation", "disruption",
             "travelled", "chart_prepared"}
    for k in sorted(set(p) - known):
        warnings.append(f"ignored unknown field '{k}'")

    # fare
    fare = None
    try:
        fare = float(p.get("fare", ""))
        if not math.isfinite(fare) or not fare > 0:
            problems.append(f"fare: must be a positive finite amount, got {fare}")
            fare = None
        elif fare > 50000:
            warnings.append(f"fare ₹{fare:.0f} is unusually high — double-check")
    except (TypeError, ValueError):
        problems.append("fare: missing or not a number — the per-passenger "
                        "fare paid is required")

    # coded fields ("cls" or "class" accepted; an explicit null falls through)
    cls_raw = p.get("cls") or p.get("class")
    cls = _canon(cls_raw, CLASS_ALIASES, VALID_CLASSES) if cls_raw else None
    if cls is None:
        if cls_raw and str(cls_raw).strip().upper() in AMBIGUOUS:
            problems.append(AMBIGUOUS[str(cls_raw).strip().upper()])
        else:
            problems.append(f"class: {'missing' if not cls_raw else f'unknown value {cls_raw!r}'}"
                            f" — expected one of {sorted(VALID_CLASSES)} "
                            "(flat charges differ by class)")

    def coded(name, aliases, valid, default):
        raw = p.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        if str(raw).strip().upper() in AMBIGUOUS:
            problems.append(AMBIGUOUS[str(raw).strip().upper()])
            return None
        v = _canon(raw, aliases, valid)
        if v is None:
            problems.append(f"{name}: unknown value {raw!r} — expected one of "
                            f"{sorted(valid)}")
        return v

    quota = coded("quota", QUOTA_ALIASES, VALID_QUOTAS, "GN")
    status = coded("status", STATUS_ALIASES, VALID_STATUS, "CNF")
    channel = coded("channel", CHANNEL_ALIASES, {"E", "C"}, "E")
    train_type = coded("train_type", TRAIN_ALIASES, VALID_TRAIN, "REG")
    disruption = coded("disruption", DISRUPTION_ALIASES, VALID_DISRUPTION, "NONE")

    departure = _dt(p.get("departure"), "departure", problems, warnings)
    cancellation = _dt(p.get("cancellation"), "cancellation", problems, warnings)
    if departure and cancellation and cancellation > departure \
            and disruption == "NONE":
        warnings.append("cancellation is AFTER scheduled departure — treated "
                        "as a no-show; TDR exception paths need the disruption "
                        "field if a disruption occurred")

    def boolean(name):
        raw = p.get(name, False)
        if isinstance(raw, str):   # LLM extractors emit "false" as a string
            v = raw.strip().lower()
            if v in ("true", "yes", "y", "1"):
                return True
            if v in ("", "false", "no", "n", "0"):
                return False
            problems.append(f"{name}: unknown value {raw!r} — expected true/false")
            return False
        return bool(raw)

    travelled = boolean("travelled")
    chart_prepared = boolean("chart_prepared")

    if problems:
        return ParseResult(None, problems, warnings)
    return ParseResult(
        TicketCase(fare=fare, cls=cls, quota=quota, status=status,
                   channel=channel, train_type=train_type,
                   departure=departure, cancellation=cancellation,
                   disruption=disruption, travelled=travelled,
                   chart_prepared=chart_prepared),
        [], warnings)
