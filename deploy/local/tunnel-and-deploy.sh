#!/bin/sh
# Run the Cloudflare quick tunnel to the local server and, whenever it comes
# up with a new address, point the Vercel project at it. Started by launchd
# (com.firsthand.tunnel); restarts the tunnel if it exits.
set -u
REPO=/Users/aniket.bhattacharjee/humanizer
export PATH="$HOME/.local/node/bin:$HOME/bin:/usr/bin:/bin:/usr/sbin:/sbin"
LOG=$HOME/Library/Logs/firsthand/tunnel.log
URLFILE=$REPO/deploy/local/tunnel.url
while true; do
  : > "$LOG.current"
  cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate --protocol http2 >> "$LOG.current" 2>&1 &
  CF=$!
  URL=""
  for i in $(seq 1 60); do
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG.current" | head -1)
    [ -n "$URL" ] && break
    sleep 1
  done
  if [ -n "$URL" ]; then
    echo "$(date -u +%FT%TZ) tunnel up: $URL" >> "$LOG"
    if [ "$(cat "$URLFILE" 2>/dev/null)" != "$URL" ]; then
      echo "$URL" > "$URLFILE"
      "$REPO/deploy/vercel/make-site.sh" "$URL" >> "$LOG" 2>&1
      (cd "$REPO/deploy/vercel/site" && vercel deploy --prod --yes --scope asha22 >> "$LOG" 2>&1) && echo "$(date -u +%FT%TZ) vercel redeployed -> $URL" >> "$LOG"
    fi
  else
    echo "$(date -u +%FT%TZ) tunnel gave no address in 60 s" >> "$LOG"
  fi
  wait $CF
  echo "$(date -u +%FT%TZ) tunnel exited; restarting in 5 s" >> "$LOG"
  sleep 5
done
