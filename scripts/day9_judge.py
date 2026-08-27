"""Day 9: validator-vs-judge faithfulness study on a ground-truth set.

Runs the deterministic validator (explain.validate_prose) and an
independent LLM judge (judge.judge_faithfulness) over prose we corrupted
in known ways, then reports each against ground truth and the Cohen's
kappa between them. The judge is provider-dependent; with no provider the
script still reports the (deterministic) validator column and the expected
behavior, and skips kappa.

    .venv/bin/python scripts/day9_judge.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entitled.agent import answer_payload
from entitled.explain import validate_prose
from entitled.judge import (build_faithfulness_samples, judge_faithfulness,
                            cohen_kappa)

# rebuild the Answer/case each sample refers to, for validate_prose context
GOLDEN = {c["id"]: c for c in json.loads((ROOT / "data" / "golden.json").read_text())["cases"]}
from entitled.parser import parse_case

samples = build_faithfulness_samples()
rows = []
for s in samples:
    c = GOLDEN[s["case_id"]]
    ans = answer_payload(c["payload"])
    case = parse_case(c["payload"]).case
    v_pass, _ = validate_prose(s["candidate"], ans, case)
    j_faithful, jdiag = judge_faithfulness(s["reference"], s["candidate"])
    rows.append({**s, "validator_pass": v_pass,
                 "judge_faithful": j_faithful,
                 "judge_reason": jdiag.get("reason", "")})

n = len(rows)
gt = [r["gt_faithful"] for r in rows]
val = [r["validator_pass"] for r in rows]

val_acc = sum(v == g for v, g in zip(val, gt)) / n
val_behaves = all(r["validator_pass"] == r["expect_validator_pass"] for r in rows)

print("Day 9 — faithfulness: deterministic validator vs LLM judge")
print("=" * 60)
print(f"samples                 {n}  ({sum(gt)} faithful, {n - sum(gt)} corrupted)")
print(f"validator accuracy      {val_acc:.1%} vs ground truth")
print(f"validator as-documented {val_behaves}  (catches all but value-swap)")

judged = [r for r in rows if r["judge_faithful"] is not None]
if judged:
    jg = [r["gt_faithful"] for r in judged]
    jj = [r["judge_faithful"] for r in judged]
    jv = [r["validator_pass"] for r in judged]
    judge_acc = sum(j == g for j, g in zip(jj, jg)) / len(judged)
    kappa = cohen_kappa(jv, jj)
    swaps = [r for r in judged if r["kind"] == "value_swap"]
    swap_caught = sum(not r["judge_faithful"] for r in swaps)
    print(f"judge coverage          {len(judged)}/{n} (provider available)")
    print(f"judge accuracy          {judge_acc:.1%} vs ground truth")
    print(f"kappa(validator, judge) {kappa:.3f}")
    print(f"value-swaps caught      judge {swap_caught}/{len(swaps)}, "
          f"validator {sum(not r['validator_pass'] for r in swaps)}/{len(swaps)}")
else:
    print("judge coverage          0/%d  (no LLM provider — kappa skipped)" % n)

out = ROOT / "reports" / "judge_eval.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"n": n, "validator_accuracy": val_acc,
                           "validator_as_documented": val_behaves,
                           "rows": rows}, indent=2))
print(f"\nsaved -> {out.relative_to(ROOT)}")
