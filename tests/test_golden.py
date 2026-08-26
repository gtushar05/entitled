"""The frozen golden set is the project's regression firewall.

- The file's own sha256 must match its recorded hash (no silent drift;
  a deliberate change means rerunning build_golden.py, which restamps it).
- The offline eval must be a PERFECT pass: 100% exact match, zero
  dangerous computes. The golden values ARE the deterministic spine's
  contract — any drift is a regression, by definition.
- Structural guards: enough cases, enough traps, ids unique, every
  must_cite id exists in the clause store.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.evalsuite import load_golden, score_offline
from entitled.retrieval import ClauseRetriever

ROOT = Path(__file__).resolve().parents[1]
DOC = json.loads((ROOT / "data" / "golden.json").read_text())


def test_golden_hash_matches_recorded():
    payload = json.dumps(DOC["cases"], indent=2, ensure_ascii=False,
                         sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    assert digest == DOC["meta"]["sha256"], \
        "golden set changed without rebuild — run scripts/build_golden.py"


def test_offline_eval_is_perfect():
    r = score_offline()
    assert r["exact_match"] == 1.0, r["failures"]
    assert r["dangerous_compute_count"] == 0, r["dangerous_ids"]
    assert r["refund_exactness"] == 1.0
    assert r["citation_accuracy"] == 1.0


def test_trap_coverage_is_substantial():
    cases = load_golden()
    assert len(cases) >= 60
    assert sum(c["trap"] for c in cases) >= 15


def test_ids_unique():
    cases = load_golden()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))


def test_every_must_cite_id_exists_in_store():
    r = ClauseRetriever(use_dense=False)
    for c in load_golden():
        for cid in c["expect"]["must_cite"]:
            assert r.by_id(cid) is not None, f"{c['id']} cites missing {cid}"


def test_categories_cover_all_regimes():
    cats = {c["category"] for c in load_golden()}
    assert {"R2015-CNF", "VB2026", "APR2026", "DISRUPTION",
            "NEEDS_INFO"}.issubset(cats)
