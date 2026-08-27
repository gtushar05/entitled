"""Day 11: ablations.

    .venv/bin/python scripts/day11_ablation.py            # retrieval (offline) only
    .venv/bin/python scripts/day11_ablation.py --llm-only # + live calculator-necessity baseline

Part B (retrieval) is offline and always runs. Part A (LLM-only baseline)
is provider-dependent and runs on the trap subset only to stay bounded.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entitled.ablation import retrieval_ablation, score_llm_only
from entitled.retrieval import ClauseRetriever

report = {"retrieval": retrieval_ablation()}

print("Part B — retrieval ablation (recall@3, offline)")
print("=" * 50)
ra = report["retrieval"]
print(f"dense backend: {ra['dense_backend']}  |  "
      f"{ra['n_lexical']} lexical + {ra['n_paraphrase']} paraphrase queries")
print(f"  {'mode':<7} {'overall':>8} {'lexical':>8} {'paraphrase':>11}")
for mode in ("bm25", "dense", "hybrid"):
    m = ra["modes"][mode]
    print(f"  {mode:<7} {m['recall_at_3']:>7.0%} {m['lexical_recall']:>8.0%} "
          f"{m['paraphrase_recall']:>11.0%}")
print("  finding: BM25 carries lexical queries; dense is the only mode to "
      "hit a paraphrase; RRF fusion ~ BM25 on this 31-clause corpus.")

if "--llm-only" in sys.argv:
    cases = json.loads((ROOT / "data" / "golden.json").read_text())["cases"]
    traps = [c for c in cases if c["trap"]]
    print(f"\nPart A — calculator necessity (LLM-only, no calculator)")
    print("=" * 50)
    print(f"running on {len([c for c in traps if c['expect']['outcome'] in ('COMPUTED','NO_REFUND','ESCALATE')])} "
          "computational trap cases (live)...", flush=True)
    # BM25-only retriever keeps this offline-except-for-the-LLM
    llm_report = score_llm_only(traps, ClauseRetriever(use_dense=False))
    report["llm_only_on_traps"] = llm_report
    print(f"  outcome accuracy   {llm_report['outcome_accuracy']:.0%}")
    print(f"  full (outcome+₹)   {llm_report['full_accuracy']:.0%}   "
          f"[verified system: 100%]")
    print(f"  trap full accuracy {llm_report['trap_accuracy']:.0%}")
    print(f"  UNSAFE ₹ outputs   {llm_report['unsafe_rupee_outputs']}   "
          "(wrong/fabricated amounts the rules don't support)")

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "ablation.json").write_text(json.dumps(report, indent=2))
print(f"\nsaved -> reports/ablation.json")
