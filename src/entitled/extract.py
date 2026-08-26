"""Natural language -> candidate field payload (the LLM's ONLY writing
role on the input side). The LLM proposes a JSON payload; parser.py
disposes. It never sees the rules and never computes anything — a wrong
extraction surfaces as a NEEDS_INFO question or a wrong *input* the user
can see echoed back, never as silently wrong math.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Callable

from .llm import complete

SYSTEM = """You extract Indian Railways ticket-cancellation facts into JSON.
Output ONLY a JSON object — no prose, no markdown fences. Keys (omit any
the user did not state; NEVER guess):
  fare (number, per passenger), cls (1A/EC/2A/FC/3A/CC/3E/SL/2S),
  quota (GN/TQ/PT), status (CNF/RAC/WL), channel (E=online, C=counter),
  train_type (REG, VBS=Vande Bharat Sleeper, AB2=Amrit Bharat 2.0),
  departure ("YYYY-MM-DD HH:MM"), cancellation ("YYYY-MM-DD HH:MM"),
  disruption (NONE/TRAIN_CANCELLED/DELAY_GT_3H),
  travelled (bool), chart_prepared (bool)
Resolve relative times ("tomorrow 6pm", "cancelled this morning") against
the reference time you are given. If the user gives a duration ("2 days
before departure"), derive the missing datetime from the other one.
If a value is ambiguous (plain "Amrit Bharat", "unreserved"), copy the
user's words verbatim as the value — validation will ask about it."""


def _find_json(text: str) -> dict | None:
    """First balanced {...} block in the reply (fence-tolerant)."""
    m = re.search(r"\{", text)
    if not m:
        return None
    depth, start = 0, m.start()
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def extract_fields(question: str, now: datetime,
                   llm: Callable[[str, str], tuple] = complete
                   ) -> tuple[dict | None, dict]:
    """Returns (payload, diagnostics). payload=None: no provider or
    unparseable reply — the caller falls back to structured input."""
    prompt = (f"Reference time (now): {now:%Y-%m-%d %H:%M} IST\n"
              f"User's situation:\n{question}\n\nJSON:")
    text, diag = llm(prompt, SYSTEM)
    if text is None:
        diag["extracted"] = False
        return None, diag
    payload = _find_json(text)
    diag["extracted"] = payload is not None
    if payload is None:
        diag["raw_head"] = text[:120]
    return payload, diag
