"""Day 6 demo: the deterministic agent spine, end to end, zero LLM.

Five representative cases through answer_payload — the exact answers the
Day-7 LLM layer will rephrase (and the kill-switch will fall back to).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.agent import answer_payload, answer_party

CASES = [
    ("Everyday case — 3A, 24h before, pre-Apr-2026",
     {"fare": 1500, "cls": "3AC", "departure": "2026-03-20 18:00",
      "cancellation": "2026-03-19 18:00"}),
    ("Vande Bharat Sleeper, 4 days early (Jan-2026 regime, verified)",
     {"fare": 2400, "cls": "2A", "train_type": "VB Sleeper",
      "departure": "2026-06-10 18:00", "cancellation": "2026-06-06 12:00"}),
    ("Apr-2026 regime, 5 days early — flat-decided: pre-committed ESCALATE",
     {"fare": 1500, "cls": "SL", "departure": "2026-06-10 18:00",
      "cancellation": "2026-06-05 18:00"}),
    ("Train cancelled by railways — full refund, any regime",
     {"fare": 900, "cls": "SL", "disruption": "train cancelled",
      "departure": "2026-03-20 18:00", "cancellation": "2026-03-20 19:00"}),
    ("Incomplete intake — agent asks, computes nothing",
     {"cls": "sleeper", "departure": "2026-03-20"}),
]

for title, payload in CASES:
    print("=" * 72)
    print(title)
    print("-" * 72)
    a = answer_payload(payload)
    print(a.explanation)
    print(f"[trace] {' | '.join(a.trace)}")

print("=" * 72)
print("Party of 3 — GN + Tatkal + WL, partial cancellation")
print("-" * 72)
base = {"fare": 1000, "cls": "3A", "departure": "2026-03-20 18:00",
        "cancellation": "2026-03-19 18:00"}
party = answer_party([dict(base), dict(base, quota="Tatkal"),
                      dict(base, status="waitlisted")])
print(party["explanation"])
