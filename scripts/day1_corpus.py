import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entitled.corpus import fetch_all

print("=== Entitled Day 1: corpus acquisition with provenance ===")
m = fetch_all()
ok = sum(1 for d in m["documents"].values() if d.get("ok"))
print(f"\n{ok}/{len(m['documents'])} documents cached -> corpus/raw/  manifest -> corpus/MANIFEST.json")
