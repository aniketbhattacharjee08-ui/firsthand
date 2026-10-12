#!/bin/sh
# Run the whole product on this Mac for testing, the way vervly.app runs it:
# the paywall with the free guest allowance, password sign-in, Your voice,
# the local MLX models, the free detector. No Stripe, no remote model server,
# no GPTZero key (the product never spends one), so nothing costs anything.
#
#   deploy/local/localhost.sh            # http://127.0.0.1:8800
#   PORT=8801 deploy/local/localhost.sh
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PORT=${PORT:-8800}
SECRET_FILE="$ROOT/deploy/local/localhost.secret"
if [ ! -s "$SECRET_FILE" ]; then
  "$ROOT/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(48))' > "$SECRET_FILE"
  chmod 600 "$SECRET_FILE"
fi
export LONGHAND_PRODUCT_NAME=Vervly
export LONGHAND_PAYWALL=1
export LONGHAND_SECRET=$(cat "$SECRET_FILE")
export LONGHAND_DB="$ROOT/data/cache/billing-localhost.db"
export LONGHAND_LOG_DIR="$ROOT/data/cache/localhost-logs"
export LONGHAND_PUBLIC_URL="http://127.0.0.1:$PORT"
export HUMANIZER_PUBLIC_URL="http://127.0.0.1:$PORT"
export LONGHAND_DEV_LINKS=1
export LONGHAND_GUESTS=1
export LONGHAND_GUEST_CREDITS=${LONGHAND_GUEST_CREDITS:-1000}
export LONGHAND_FREE_CREDITS=${LONGHAND_FREE_CREDITS:-1000}
export LONGHAND_WORDS_PER_CREDIT=1
export LONGHAND_MAX_INFLIGHT=1
export HUMANIZER_AUTH=1
export HUMANIZER_PUBLIC_APP=1
export HUMANIZER_AUTH_DB="$ROOT/data/auth.sqlite"
export HUMANIZER_VOICE_DB="$ROOT/data/voice.sqlite"
export HUMANIZER_ASSUME_AI=1
export HUMANIZER_PROXY=surrogate
export HUMANIZER_YARDSTICK=surrogate
export HUMANIZER_MAX_RESIDENT_LLM=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HUMANIZER_LLM_URL HUMANIZER_LLM_API_KEY GPTZERO_API_KEY STRIPE_SECRET_KEY STRIPE_WEBHOOK_SECRET STRIPE_PRICES STRIPE_PLANS HUMANIZER_MAINTENANCE
mkdir -p "$LONGHAND_LOG_DIR"
exec "$ROOT/.venv/bin/python" -W ignore -m humanizer.billing serve --host 127.0.0.1 --port "$PORT"
