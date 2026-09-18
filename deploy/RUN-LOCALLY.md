# Run Firsthand on your own computer

Firsthand is a Python server plus a web page. Anyone with an Apple Silicon Mac
can run the whole thing locally; nothing phones home except the model
downloads on first install and, if they choose to add one, their own GPTZero
key.

## Requirements

- A Mac with an M1 or later chip. The rewriter uses MLX, which is Apple only.
  Intel Macs, Windows and Linux can run the measurement and the rule-based
  engine but not the LLM rewriter.
- 16 GB of memory minimum, 24 GB comfortable. About 15 GB of disk.
- Python 3.9 or newer.

## Install

```bash
git clone <repo-url> firsthand
cd firsthand
sh deploy/install-local.sh
```

The script creates a virtual environment, installs the dependencies,
downloads the two Qwen2.5-7B checkpoints (Apache 2.0, about 8 GB) and makes
an admin account (set `FIRSTHAND_ADMIN_EMAIL=you@example.com` first, or it
uses a placeholder you can reset later with
`.venv/bin/humanizer account master <email>`).

The trained adapter that makes the 7B base pass detectors ships in the repo at
`data/adapters/hip7b-r4-it200`. The free detector, a 480 MB model trained on
GPTZero verdicts, does not fit in git. The owner shares it as
`firsthand-weights.zip`; unzip it at the repo root so that
`data/cache/models/gptzero-surrogate/` exists. Without it the app still
rewrites, but shows no free reading; a user's own GPTZero key still works.

## Run

```bash
.venv/bin/humanizer serve --port 8000
```

Open http://127.0.0.1:8000, sign in, paste a paragraph. The first rewrite
loads the model (about 30 s), then a paragraph takes 45 to 90 s.

## What is not included

- Google and Apple sign-in need the operator's own keys (`deploy/README.md`).
- The credit paywall (`humanizer.billing`) is off; the local copy is free.
