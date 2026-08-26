"""Reconcile the hand-computed golden set against the live agent.

Every disagreement is a finding: either a hand-arithmetic slip in
build_golden.py (fix the literal) or a real calculator/agent bug (fix the
code). This runs OFFLINE on the structured payloads — no LLM, no network.
Prints a per-case diff; exits non-zero if anything disagrees.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entitled.agent import answer_payload

doc = json.loads((ROOT / "data" / "golden.json").read_text())
cases = doc["cases"]


def bound_ids(ans):
    return {c.clause_id for c in ans.citations if c.bound}


mismatches = []
for c in cases:
    ans = answer_payload(c["payload"])
    e = c["expect"]
    diffs = []
    if ans.outcome != e["outcome"]:
        diffs.append(f"outcome {ans.outcome!r} != {e['outcome']!r}")
    if e["refund"] is not None and ans.refund != e["refund"]:
        diffs.append(f"refund {ans.refund} != {e['refund']}")
    if e["charge"] is not None and ans.charge != e["charge"]:
        diffs.append(f"charge {ans.charge} != {e['charge']}")
    if e["regime"] is not None and ans.regime != e["regime"]:
        diffs.append(f"regime {ans.regime!r} != {e['regime']!r}")
    if e["verified"] is not None and ans.verified != e["verified"]:
        diffs.append(f"verified {ans.verified} != {e['verified']}")
    missing = set(e["must_cite"]) - bound_ids(ans)
    if missing:
        diffs.append(f"missing bound cites {sorted(missing)} "
                     f"(have {sorted(bound_ids(ans))})")
    blob = (ans.explanation + " " + " ".join(ans.questions)).lower()
    for sub in e["must_ask"]:
        if sub.lower() not in blob:
            diffs.append(f"answer missing ask-substring {sub!r}")
    if diffs:
        mismatches.append((c["id"], diffs))

if mismatches:
    print(f"MISMATCHES: {len(mismatches)}/{len(cases)}\n")
    for cid, diffs in mismatches:
        print(f"  {cid}")
        for d in diffs:
            print(f"      - {d}")
    sys.exit(1)
print(f"OK: all {len(cases)} golden cases reconcile with the live agent.")
