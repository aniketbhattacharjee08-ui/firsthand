#!/bin/sh
# Run Firsthand on your own Mac. Needs Apple Silicon (M1 or later), about
# 16 GB of memory (24 GB is comfortable), 15 GB of disk and Python 3.9+.
#
#   git clone <repo-url> firsthand && cd firsthand
#   sh deploy/install-local.sh            # ~10 minutes, mostly downloads
#   .venv/bin/humanizer serve --port 8000 # then open http://127.0.0.1:8000
#
# The rewriter weights (Qwen2.5-7B base and instruct, Apache 2.0) download
# from Hugging Face on first run of this script. The trained adapter ships in
# the repo. The free detector (the GPTZero surrogate) is 480 MB and lives in
# data/cache/models/gptzero-surrogate; unzip firsthand-weights.zip at the repo
# root to add it, or the app runs without a free reading.
set -eu
cd "$(dirname "$0")/.."
if [ "$(uname -m)" != "arm64" ] || [ "$(uname -s)" != "Darwin" ]; then
  echo "Firsthand's rewriter runs on Apple Silicon Macs only (MLX)."; exit 1
fi
PY=$(command -v python3.11 || command -v python3.10 || command -v python3.9 || command -v python3)
echo "== Python: $PY ($($PY --version))"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
echo "== installing (fastapi, torch, transformers, mlx-lm; a few minutes)"
.venv/bin/python -m pip install -q -e ".[api,detectors]" mlx-lm python-multipart
echo "== downloading the rewriter checkpoints (about 8 GB)"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for m in ("mlx-community/Qwen2.5-7B-4bit", "mlx-community/Qwen2.5-7B-Instruct-4bit"):
    print("  ", m); snapshot_download(m)
PY
mkdir -p data/cache/models data/reference
if [ -d data/cache/models/gptzero-surrogate ]; then echo "== free detector present"; else
  echo "== free detector missing: unzip firsthand-weights.zip here to add data/cache/models/gptzero-surrogate"; fi
[ -d data/adapters/hip7b-r4-it200 ] && echo "== adapter present: data/adapters/hip7b-r4-it200" || echo "== adapter missing (pull the repo again)"
echo "== creating your admin account"
.venv/bin/humanizer account master "${FIRSTHAND_ADMIN_EMAIL:-you@example.com}" || true
echo
echo "Done. Start it with:   .venv/bin/humanizer serve --port 8000"
echo "then open http://127.0.0.1:8000 and sign in with the account above."
