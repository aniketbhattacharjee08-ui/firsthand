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
# `routes` rather than `rewrites` so the proxy can stamp every forwarded
# request with the origin secret (a Vercel project env var, ORIGIN_SECRET,
# which go.sh sets from LONGHAND_ORIGIN_SECRET). The API believes the
# visitor's address only on requests carrying it, and refuses the rest.
cat > "$SITE/vercel.json" <<JSON
{
  "cleanUrls": false,
  "routes": [
    {
      "src": "/(.*)",
      "dest": "${API}/\$1",
      "headers": {
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin"
      },
      "transforms": [
        {
          "type": "request.headers",
          "op": "set",
          "target": { "key": "x-origin-secret" },
          "args": "\$ORIGIN_SECRET",
          "env": ["ORIGIN_SECRET"]
        }
      ]
    }
  ]
}
JSON
# Vercel wants at least one static file in a project; the server answers "/"
# through the rewrite, so this placeholder is never served.
printf 'Vervly\n' > "$SITE/.placeholder"
echo "site ready: $SITE -> $API"
