#!/bin/sh
# Copy the weights that are NOT on the Hugging Face hub from the dev machine
# to the host. Run on the dev machine:
#   deploy/sync-weights.sh longhand@host.example.com:/Users/longhand/humanizer
# Everything else (Qwen2.5-7B-4bit, desklib) downloads on first use or via
# deploy/warm.py.
set -eu
DEST=${1:?usage: sync-weights.sh user@host:/path/to/humanizer}
HERE=$(cd "$(dirname "$0")/.." && pwd)
rsync -avz --progress \
  "$HERE/data/cache/models/gptzero-surrogate/" "$DEST/data/cache/models/gptzero-surrogate/"
rsync -avz --progress \
  "$HERE/data/cache/models/qwen3-4b-instruct-4bit/" "$DEST/data/cache/models/qwen3-4b-instruct-4bit/"
rsync -avz --progress \
  "$HERE/data/adapters/" "$DEST/data/adapters/"   # includes hip7b-r4-it200 (the shipped adapter)
rsync -avz --progress \
  "$HERE/data/reference/" "$DEST/data/reference/"
echo done
