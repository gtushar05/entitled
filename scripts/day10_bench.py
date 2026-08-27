"""Day 10: run the provider frontier on a small curated subset, live.

    .venv/bin/python scripts/day10_bench.py                 # haiku sonnet
    .venv/bin/python scripts/day10_bench.py haiku            # one tier
    ANTHROPIC_API_KEY=... .venv/bin/python scripts/day10_bench.py haiku sonnet opus

Writes reports/provider_frontier.json and reports/frontier.svg. Kept to
~10 cases x N tiers so a CLI run stays bounded; use an API key for the
full 70-case sweep across all three tiers.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entitled.bench import benchmark
from entitled.llm import complete

# curated subset spanning outcomes/regimes; all self-contained questions
SUBSET = ["r2015-3a-36h", "r2015-3a-4h", "r2015-sl-60h", "vb-2a-2400-102h",
          "vb-8h-exact", "apr-3a-36h", "apr-3a-102h", "r2015-cancelled-e-sl-900",
          "ni-unreserved", "r2015-wl-sl-900-e-20h"]

TIERS = sys.argv[1:] or ["haiku", "sonnet"]
by_id = {c["id"]: c for c in json.loads((ROOT / "data" / "golden.json").read_text())["cases"]}
cases = [by_id[i] for i in SUBSET]

results = []
for tier in TIERS:
    print(f"benchmarking {tier} on {len(cases)} cases ...", flush=True)
    base = (lambda t: (lambda p, s=None: complete(p, s, model=t)))(tier)
    r = benchmark(tier, cases, base)
    results.append(r)
    print(f"  extraction {r['extraction_accuracy']:.0%} | "
          f"faithful {r['polish_faithful_rate']} | "
          f"p95 {r['latency_p95']:.1f}s | "
          f"${r['projected_cost_per_1k']:.2f}/1k queries", flush=True)

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "provider_frontier.json").write_text(
    json.dumps({"subset": SUBSET, "tiers": results}, indent=2))


def svg_frontier(results: list[dict]) -> str:
    """Minimal dependency-free scatter: x=p95 latency, y=extraction acc,
    bubble label = tier + $/1k."""
    W, H, pad = 520, 320, 60
    xs = [r["latency_p95"] for r in results] or [1]
    xmax = max(xs) * 1.25 or 1
    def px(x): return pad + (x / xmax) * (W - 2 * pad)
    def py(acc): return H - pad - acc * (H - 2 * pad)   # acc in [0,1]
    dots, labels = [], []
    palette = {"haiku": "#2a9d8f", "sonnet": "#e76f51", "opus": "#8e44ad"}
    for r in results:
        cx, cy = px(r["latency_p95"]), py(r["extraction_accuracy"] or 0)
        col = palette.get(r["tier"], "#555")
        dots.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="8" '
                    f'fill="{col}" opacity="0.85"/>')
        labels.append(f'<text x="{cx + 12:.0f}" y="{cy + 4:.0f}" '
                      f'font-size="12" fill="{col}">{r["tier"]} '
                      f'(${r["projected_cost_per_1k"]:.2f}/1k)</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
<text x="{W/2:.0f}" y="24" text-anchor="middle" font-size="15" font-weight="bold">Entitled — provider frontier</text>
<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#333"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#333"/>
<text x="{W/2:.0f}" y="{H-20}" text-anchor="middle" font-size="12">p95 latency (s) →</text>
<text x="20" y="{H/2:.0f}" text-anchor="middle" font-size="12" transform="rotate(-90 20 {H/2:.0f})">extraction accuracy →</text>
{''.join(dots)}
{''.join(labels)}
</svg>'''


(ROOT / "reports" / "frontier.svg").write_text(svg_frontier(results))
print(f"\nsaved -> reports/provider_frontier.json, reports/frontier.svg")
