"""Day 7 demo: natural language in, validated cited answer out.

Uses the real provider chain (ANTHROPIC_API_KEY -> claude CLI) when one
is available; otherwise shows the graceful no-extractor degradation.
Run with a key to see live extraction + polish:
    ANTHROPIC_API_KEY=... .venv/bin/python scripts/day7_assistant_demo.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.assistant import answer_question

NOW = datetime(2026, 3, 18, 12, 0)

QUESTIONS = [
    "I booked a 3AC ticket for Rs 1500 on a regular express leaving 20th "
    "March at 6pm. I cancelled it online this morning at 10. What do I "
    "get back?",
    "Vande Bharat sleeper, 2A, fare 2400, departs June 10 at 6pm and I "
    "want to cancel today. How much do I lose?",
    "My waitlisted sleeper ticket (fare 900) — train leaves tomorrow 8am, "
    "cancelling now.",
    "The railways cancelled my train! SL ticket, 900 rupees, was leaving "
    "tonight 8pm.",
]

for q in QUESTIONS:
    print("=" * 72)
    print("Q:", q)
    print("-" * 72)
    out = answer_question(q, now=NOW)
    print(out["text"])
    print(f"[mode: {out['mode']} | extract: "
          f"{out['diagnostics']['extract'].get('provider')}]")
