# Entitled — a rules-grounded agent for Indian Railways refunds

**Describe your ticket situation; get the exact refund you're owed — computed
deterministically from the governing rules, cited to the clause, version-aware
by cancellation date and train type.**

> 🚧 Day 1 of a 14-day build. Why version-aware? Because the refund rules
> changed twice in 2026 — and as cached in this repo's own corpus (with
> hashes and fetch dates), **IRCTC's official rules page still displays the
> superseded 2015 tiers.** The official sources currently disagree with each
> other; this agent cites the gazette.

## Day 1 state

- Corpus acquired with provenance: 9 documents cached, hashed, and dated in
  [corpus/MANIFEST.json](corpus/MANIFEST.json) — including the 29-page scanned
  2015 gazette (OCR'd, 11.5K words) and **Railway Board CC 08/2026 carrying
  G.S.R. 41(E)**, from which the Vande Bharat Sleeper / Amrit Bharat II regime
  is **verified from primary text**: >72h → 25% charge · 72–8h → 50% · <8h →
  no refund (rule 6(4), with the 6(5) override and the rule-8 proviso).
- The April-2026 all-trains regime (72/24/8) is confirmed by multiple
  secondary sources; its primary circular is not yet in hand → per the
  pre-committed verification rule, flat-minimum branches remain
  **UNVERIFIED → ESCALATE** until it is (Day 2 hunts the 2026 CC index).

## The plan

Deterministic regime-aware refund calculator (the LLM never does arithmetic)
→ citation-grounded retrieval over clause-level chunks → agent loop with
escalation on out-of-scope cases → frozen ~75-case golden set (including
15 trap cases where official sources are currently wrong) → cost/latency
across providers → capped Gradio demo. Sibling project:
[Incremental](https://github.com/gtushar05/incremental) — same doctrine,
different domain.
