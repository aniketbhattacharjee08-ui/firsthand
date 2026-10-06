#!/bin/sh
# Create Vervly's Stripe products, prices and webhook endpoint, and write the
# resulting ids into deploy/.env. Idempotent: products are looked up by name
# before being created. Needs the account's secret key in STRIPE_SECRET_KEY
# (sk_test_... first; rerun with sk_live_... to go live).
#
#   STRIPE_SECRET_KEY=sk_test_... deploy/cloud/stripe-setup.sh
#
# Plan economics (2026-10-05): a 500-word rewrite costs about 2 cents of GPU
# plus up to 4 cents of idle time on the scale-to-zero server, so 1,000 words
# costs at most about 12 cents. The allowances below keep every plan above
# cost even for a user who spends the whole allowance every month.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENV_FILE="$ROOT/deploy/.env"
: "${STRIPE_SECRET_KEY:?set STRIPE_SECRET_KEY (sk_test_... or sk_live_...)}"
PUBLIC_URL=${PUBLIC_URL:-$(/usr/bin/grep -E '^LONGHAND_PUBLIC_URL=' "$ENV_FILE" | cut -d= -f2-)}
API=https://api.stripe.com/v1
auth() { curl -fsS -u "$STRIPE_SECRET_KEY:" "$@"; }
jsonget() { python3 -c 'import sys,json; d=json.load(sys.stdin); print(eval(sys.argv[1], {}, {"d": d}))' "$1"; }
envset() {
  if /usr/bin/grep -qE "^$1=" "$ENV_FILE"; then
    tmp=$(mktemp); awk -v k="$1" -v v="$2" 'BEGIN{FS=OFS="="} $1==k{$0=k"="v} {print}' "$ENV_FILE" > "$tmp" && /bin/cat "$tmp" > "$ENV_FILE" && rm -f "$tmp"
  else printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"; fi
}

# name | product name | unit amount (cents) | interval (month|year|lifetime) | allowance credits | label
PLANS="monthly|Vervly Monthly|999|month|300|\$9.99 a month
yearly|Vervly Yearly|7999|year|300|\$79.99 a year
lifetime|Vervly Lifetime|29900|lifetime|200|\$299 once"
# one-time top-ups: name | product name | cents | credits | label
PACKS="topup100|Vervly 10,000 words|499|100|\$4.99 for 10,000 words"

product_id() {  # product_id "<name>"
  auth "$API/products/search" --get --data-urlencode "query=name:'$1' AND active:'true'" | jsonget "d['data'][0]['id'] if d['data'] else ''"
}
# Managed Payments (Stripe as merchant of record, on for this account) refuses
# any Checkout line item whose product has no eligible tax code. Vervly is a
# cloud AI writing service sold to individuals, so every product carries
# "AI as a Service - cloud based - personal use". Existing products missing
# the code are patched, so rerunning the script repairs an older catalogue.
TAX_CODE=${STRIPE_TAX_CODE:-txcd_10105001}
ensure_product() {
  id=$(product_id "$1")
  if [ -n "$id" ]; then
    have=$(auth "$API/products/$id" | jsonget "d.get('tax_code') or ''")
    [ "$have" = "$TAX_CODE" ] || auth "$API/products/$id" -d "tax_code=$TAX_CODE" >/dev/null
  else
    id=$(auth "$API/products" -d "name=$1" -d "tax_code=$TAX_CODE" | jsonget "d['id']")
  fi
  echo "$id"
}
ensure_price() {  # ensure_price product cents interval(month|year|'' for one-time) lookup_key
  existing=$(auth "$API/prices" --get -d "lookup_keys[]=$4" -d active=true | jsonget "d['data'][0]['id'] if d['data'] else ''")
  if [ -n "$existing" ]; then echo "$existing"; return; fi
  if [ -n "$3" ]; then
    auth "$API/prices" -d "product=$1" -d "unit_amount=$2" -d currency=usd -d "recurring[interval]=$3" -d "lookup_key=$4" | jsonget "d['id']"
  else
    auth "$API/prices" -d "product=$1" -d "unit_amount=$2" -d currency=usd -d "lookup_key=$4" | jsonget "d['id']"
  fi
}

PLAN_SPEC=""
printf '%s\n' "$PLANS" | while IFS='|' read -r name pname cents interval credits label; do
  pid=$(ensure_product "$pname")
  case "$interval" in lifetime) rint="";; *) rint="$interval";; esac
  price=$(ensure_price "$pid" "$cents" "$rint" "vervly_$name")
  printf '%s:%s:%s:%s:%s;' "$name" "$price" "$credits" "$interval" "$label"
done > /tmp/vervly.plans
PLAN_SPEC=$(sed 's/;$//' /tmp/vervly.plans); rm -f /tmp/vervly.plans
printf '%s\n' "$PACKS" | while IFS='|' read -r name pname cents credits label; do
  pid=$(ensure_product "$pname")
  price=$(ensure_price "$pid" "$cents" "" "vervly_$name")
  printf '%s:%s:%s:%s;' "$name" "$price" "$credits" "$label"
done > /tmp/vervly.packs
PACK_SPEC=$(sed 's/;$//' /tmp/vervly.packs); rm -f /tmp/vervly.packs

# Webhook endpoint for the events the billing package handles.
WH_URL="$PUBLIC_URL/api/billing/webhook"
wh=$(auth "$API/webhook_endpoints" --get -d limit=100 | jsonget "next((w['id'] for w in d['data'] if w['url']=='$WH_URL'), '')")
if [ -z "$wh" ]; then
  out=$(auth "$API/webhook_endpoints" -d "url=$WH_URL" \
    -d "enabled_events[]=checkout.session.completed" -d "enabled_events[]=checkout.session.async_payment_succeeded" \
    -d "enabled_events[]=customer.subscription.created" -d "enabled_events[]=customer.subscription.updated" \
    -d "enabled_events[]=customer.subscription.deleted" -d "enabled_events[]=invoice.paid")
  envset STRIPE_WEBHOOK_SECRET "$(printf '%s' "$out" | jsonget "d['secret']")"
  echo "webhook created: $WH_URL"
else
  echo "webhook exists: $WH_URL (secret unchanged; set STRIPE_WEBHOOK_SECRET by hand if it is missing)"
fi

envset STRIPE_SECRET_KEY "$STRIPE_SECRET_KEY"
envset STRIPE_PLANS "$PLAN_SPEC"
envset STRIPE_PRICES "$PACK_SPEC"
envset LONGHAND_PAYWALL 1
envset LONGHAND_FREE_CREDITS 10
envset LONGHAND_WORDS_PER_CREDIT 100
echo "plans:  $PLAN_SPEC"
echo "packs:  $PACK_SPEC"
echo "deploy/.env updated; now run deploy/cloud/go.sh to ship the paywall"
