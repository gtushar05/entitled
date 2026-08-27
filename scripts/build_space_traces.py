"""Freeze cached demo traces into data/cached_traces.json (the demo's
kill-switch fallback when no model is configured on the Space)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entitled.demo import build_cached_traces

traces = build_cached_traces()
out = ROOT / "data" / "cached_traces.json"
out.write_text(json.dumps(traces, indent=2, ensure_ascii=False))
print(f"wrote {len(traces)} cached traces -> {out.relative_to(ROOT)}")
