#!/usr/bin/env bash
# Assemble a self-contained HF Gradio Space in ./_space_build, mirroring the
# repo layout so `entitled` and its data resolve unchanged. The user pushes
# the staging dir to their HF Space with their own hf token (never handled
# here). The demo runs the DETERMINISTIC path only — no torch/model2vec —
# so the Space stays light; set an ANTHROPIC_API_KEY Space secret to enable
# the natural-language tab.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="_space_build"

rm -rf "$STAGE"
mkdir -p "$STAGE/src" "$STAGE/data" "$STAGE/corpus/parsed"

cp space/app.py space/requirements.txt space/README.md "$STAGE/"
cp -R src/entitled "$STAGE/src/entitled"
cp data/golden.json data/cached_traces.json "$STAGE/data/"
cp corpus/parsed/clauses.json "$STAGE/corpus/parsed/"

# drop caches / bytecode that shouldn't ship
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "*.pyc" -delete 2>/dev/null || true

echo "staged Space -> $STAGE"
echo "contents:"; find "$STAGE" -maxdepth 2 -type f | sort | sed 's/^/  /'
cat <<'EOF'

Next steps (you run these — the hf token is entered by you, never by Claude):
  1) Create a Gradio Space at https://huggingface.co/new-space (SDK: Gradio)
  2) cd _space_build && git init && git add -A && git commit -m "entitled space"
  3) git remote add origin https://huggingface.co/spaces/<user>/entitled
  4) git push -u origin main         # authenticate with your hf_ token
  (optional) add an ANTHROPIC_API_KEY secret in the Space settings to enable
             the natural-language tab; without it the demo uses cached traces.
EOF
