---
title: Entitled — Railway Refund Agent
emoji: 🚆
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Rules-grounded Indian Railways refund agent — LLM never does the math
---

# Entitled

A version-aware, **rules-grounded** agent for Indian Railways ticket refunds.

The refund arithmetic is done by a deterministic, clause-cited calculator
across three rule regimes (2015 gazette, Jan-2026 Vande-Bharat rules, the
pending Apr-2026 reform). The language model only turns your words into
structured facts and rephrases the answer — and every number it emits is
checked against the calculator by a numeric-faithfulness gate before you
see it. When the verified rules don't determine an answer, it escalates
instead of guessing.

- **Calculator tab** — runs the verified calculator live, no model needed.
- **Ask in words** — natural-language parsing (needs an `ANTHROPIC_API_KEY`
  Space secret; request-capped; falls back to cached traces otherwise).
- **Trap gallery** — the boundary/flat-minimum/escalation cases a naive
  or pure-LLM approach gets wrong.

Code, the 70-case frozen golden set, and 123 tests:
**github.com/gtushar05/entitled**
