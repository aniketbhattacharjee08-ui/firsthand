#!/bin/sh
# Container entrypoint: fetch the surrogate weights into the volume on first
# boot, then serve. The weights (480MB) ship in the GitHub release so the
# image stays small and the volume keeps them across deploys.
set -eu
WEIGHTS_URL=${VERVLY_WEIGHTS_URL:-https://github.com/aniketbhattacharjee08-ui/firsthand/releases/download/v0.1.0/firsthand-weights.zip}
mkdir -p /data/cache/models /data/logs /data/backups
if [ ! -f /data/cache/models/gptzero-surrogate/model.safetensors ]; then
  echo "[entrypoint] downloading surrogate weights"
  tmp=$(mktemp -d)
  curl -fsSL "$WEIGHTS_URL" -o "$tmp/w.zip"
  unzip -q "$tmp/w.zip" -d "$tmp/x"
  src=$(find "$tmp/x" -type d -name gptzero-surrogate | head -1)
  [ -n "$src" ] || { echo "surrogate not in bundle" >&2; exit 1; }
  mv "$src" /data/cache/models/gptzero-surrogate
  rm -rf "$tmp"
fi
# uvicorn trusts X-Forwarded-* from the platform proxy (Fly, Vercel rewrite).
export FORWARDED_ALLOW_IPS='*'
case "$(printf '%s' "${LONGHAND_PAYWALL:-0}" | tr 'A-Z' 'a-z')" in
  1|true|yes|on) exec python -m humanizer.billing --host 0.0.0.0 --port "${PORT:-8000}" ;;
esac
exec humanizer serve --host 0.0.0.0 --port "${PORT:-8000}"
