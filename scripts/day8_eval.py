"""Run the offline golden-set eval and save the report artifact.

    .venv/bin/python scripts/day8_eval.py
    .venv/bin/python scripts/day8_eval.py --extraction   # + live NL path
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entitled.evalsuite import score_offline, score_extraction, format_report

report = score_offline()
print(format_report(report))

out = ROOT / "reports" / "golden_eval.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(f"\nsaved -> {out.relative_to(ROOT)}")

if "--extraction" in sys.argv:
    print("\nrunning live extraction eval (provider-dependent)...")
    ex = score_extraction()
    print(f"outcome accuracy   {ex['outcome_accuracy']:.1%}")
    if ex["refund_exactness_given_outcome"] is not None:
        print(f"refund exactness   {ex['refund_exactness_given_outcome']:.1%} "
              "(given correct outcome)")
    print(f"no extraction      {ex['no_extraction']}")
    (ROOT / "reports" / "extraction_eval.json").write_text(json.dumps(ex, indent=2))
