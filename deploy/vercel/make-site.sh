#!/bin/sh
# Build the Vercel project directory: a static shell that proxies every path
# to the running Vervly server (a Cloudflare tunnel or, later, the rented
# Mac). Vercel gives the stable public address and TLS; the Mac does the work.
#   deploy/vercel/make-site.sh https://<tunnel-or-api-host>
set -eu
API=${1:?usage: make-site.sh https://api-host}
HERE=$(cd "$(dirname "$0")" && pwd)
SITE="$HERE/site"
mkdir -p "$SITE"
cat > "$SITE/vercel.json" <<JSON
{
  "cleanUrls": false,
  "rewrites": [
    { "source": "/(.*)", "destination": "${API}/\$1" }
  ],
  "headers": [
    { "source": "/(.*)", "headers": [
      { "key": "X-Content-Type-Options", "value": "nosniff" },
      { "key": "Referrer-Policy", "value": "strict-origin-when-cross-origin" }
    ] }
  ]
}
JSON
# Vercel wants at least one static file in a project; the server answers "/"
# through the rewrite, so this placeholder is never served.
printf 'Vervly\n' > "$SITE/.placeholder"
echo "site ready: $SITE -> $API"
