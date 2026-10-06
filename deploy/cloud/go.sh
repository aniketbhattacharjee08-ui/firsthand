#!/bin/sh
# One-shot, idempotent deploy of Vervly to Modal + Fly + Vercel.
# Re-run it any time; each stage skips when its prerequisite is missing and
# prints what is still needed. Owner-only inputs: Modal login (`modal token
# new`), Fly login (`flyctl auth login`, card on file), and, optionally,
# DOMAIN=<bought domain> for the last stage.
#
#   deploy/cloud/go.sh                 # deploy what can be deployed
#   DOMAIN=example.com deploy/cloud/go.sh
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
ENV_FILE="$ROOT/deploy/.env"
MODAL=${MODAL:-$HOME/.local/deploy-venv/bin/modal}
FLY=${FLY:-$HOME/.fly/bin/flyctl}
export PATH="$HOME/.local/node/bin:$PATH"
FLY_APP=${FLY_APP:-vervly-api}
FLY_REGION=${FLY_REGION:-sjc}
VERCEL_SCOPE=${VERCEL_SCOPE:-asha22}

envget() { /usr/bin/grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2-; }
envset() {  # envset KEY VALUE : replace or append in deploy/.env
  if /usr/bin/grep -qE "^$1=" "$ENV_FILE"; then
    tmp=$(mktemp); awk -v k="$1" -v v="$2" 'BEGIN{FS=OFS="="} $1==k{$0=k"="v} {print}' "$ENV_FILE" > "$tmp" && /bin/cat "$tmp" > "$ENV_FILE" && rm -f "$tmp"
  else printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"; fi
}
say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- 1. Modal
say "Modal (GPU model server)"
if ! "$MODAL" profile current >/dev/null 2>&1 || ! [ -s "$HOME/.modal.toml" ]; then
  echo "   needs: $MODAL token new   (browser sign-in)"
else
  if ! "$MODAL" secret list 2>/dev/null | /usr/bin/grep -q vervly-llm; then
    KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')
    "$MODAL" secret create vervly-llm VLLM_API_KEY="$KEY" >/dev/null
    envset HUMANIZER_LLM_API_KEY "$KEY"
    echo "   created secret vervly-llm"
  fi
  if [ -z "$(envget HUMANIZER_LLM_API_KEY)" ]; then
    echo "   secret vervly-llm exists but deploy/.env has no HUMANIZER_LLM_API_KEY; recreate it:"
    echo "   $MODAL secret delete vervly-llm && rerun"; exit 1
  fi
  OUT=$("$MODAL" deploy deploy/cloud/modal_vllm.py 2>&1 | tee /dev/stderr) || { echo "   modal deploy failed"; exit 1; }
  URL=$(printf '%s' "$OUT" | /usr/bin/grep -oE 'https://[a-z0-9.-]+\.modal\.run' | head -1)
  if [ -n "$URL" ]; then envset HUMANIZER_LLM_URL "$URL/v1"; echo "   model server: $URL/v1"; fi
fi

# ----------------------------------------------------------------- 2. Fly
say "Fly.io (API container)"
if ! "$FLY" auth whoami >/dev/null 2>&1; then
  echo "   needs: $FLY auth login   (interactive; add a card at https://fly.io/dashboard/personal/billing)"
else
  # `status` exits non-zero when the app does not exist; the list output is
  # a padded table, so grepping it is unreliable.
  "$FLY" status --app "$FLY_APP" >/dev/null 2>&1 || "$FLY" apps create "$FLY_APP" --org personal
  "$FLY" volumes list --app "$FLY_APP" 2>/dev/null | /usr/bin/grep -q vervly_data \
    || "$FLY" volumes create vervly_data --size 5 --region "$FLY_REGION" --app "$FLY_APP" --yes
  # Every non-empty, non-comment line of deploy/.env becomes a Fly secret.
  /usr/bin/grep -vE '^\s*(#|$)' "$ENV_FILE" | /usr/bin/grep -vE '=$' > /tmp/vervly.secrets
  "$FLY" secrets import --app "$FLY_APP" --stage < /tmp/vervly.secrets >/dev/null; rm -f /tmp/vervly.secrets
  "$FLY" deploy . --config deploy/cloud/fly.toml --dockerfile deploy/cloud/Dockerfile --app "$FLY_APP" --remote-only --yes
  API_URL="https://$FLY_APP.fly.dev"
  echo "   api: $API_URL"
  curl -fsS -m 20 "$API_URL/api/health" && echo || echo "   (health not up yet; first boot downloads 480MB of weights)"
fi

# -------------------------------------------------------------- 3. Vercel
say "Vercel (public address)"
API_URL="https://$FLY_APP.fly.dev"
if ! vercel whoami >/dev/null 2>&1; then
  echo "   needs: vercel login"
elif ! "$FLY" auth whoami >/dev/null 2>&1; then
  echo "   waiting for the Fly deploy before pointing Vercel at it"
else
  deploy/vercel/make-site.sh "$API_URL" >/dev/null
  (cd deploy/vercel/site && vercel deploy --prod --yes --scope "$VERCEL_SCOPE" 2>&1 | tail -1)
  if [ -n "${DOMAIN:-}" ]; then
    (cd deploy/vercel/site && vercel domains add "$DOMAIN" --scope "$VERCEL_SCOPE" 2>&1 | tail -2) || true
    envset LONGHAND_PUBLIC_URL "https://$DOMAIN"; envset HUMANIZER_PUBLIC_URL "https://$DOMAIN"
    /usr/bin/grep -vE '^\s*(#|$)' "$ENV_FILE" | /usr/bin/grep -vE '=$' > /tmp/vervly.secrets
    "$FLY" secrets import --app "$FLY_APP" < /tmp/vervly.secrets >/dev/null; rm -f /tmp/vervly.secrets
    echo "   public url set to https://$DOMAIN (Fly restarted with it)"
  else
    echo "   no DOMAIN given; site is at https://firsthand-navy.vercel.app until then"
  fi
fi
say "done"
