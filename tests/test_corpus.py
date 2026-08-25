import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_manifest_integrity():
    """Every cached document must exist and match its manifest hash —
    the corpus's own provenance check."""
    m = json.loads((ROOT / "corpus" / "MANIFEST.json").read_text())
    checked = 0
    for name, meta in m["documents"].items():
        if not meta.get("ok"):
            continue
        p = ROOT / "corpus" / "raw" / name
        assert p.exists(), f"missing cached doc: {name}"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["sha256"], name
        checked += 1
    assert checked >= 7

def test_gazette_ocr_exists_and_substantial():
    t = (ROOT / "corpus" / "parsed" / "refund_rules_2015_ocr.txt").read_text()
    assert len(t.split()) > 8000
    assert "MINISTRY OF RAILWAYS" in t
