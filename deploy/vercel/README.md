# Vercel in front, the Mac behind

Live since 2026-09-13 at https://firsthand-navy.vercel.app (project
`firsthand`, team `asha22`, deployed with the Vercel CLI from
`deploy/vercel/site`). Vercel holds the public address and TLS and proxies
every path to the server; the server runs on the Mac, reached through a
Cloudflare tunnel, because the rewriter needs Apple Silicon and resident
weights that no serverless platform offers.

```
browser -> https://firsthand-navy.vercel.app -> (rewrite) -> https://<tunnel>.trycloudflare.com -> 127.0.0.1:8000
```

## Runs by itself on this Mac (since 2026-09-13)

Two launch agents in `~/Library/LaunchAgents` start at login and restart on
crash: `com.firsthand.server` (the API on 127.0.0.1:8000 with
`HUMANIZER_PUBLIC_URL` set) and `com.firsthand.tunnel`, which runs
`deploy/local/tunnel-and-deploy.sh`: it starts the quick tunnel, and whenever
the address differs from `deploy/local/tunnel.url` it rewrites `vercel.json`
and redeploys, so a reboot heals itself in about a minute. Logs are in
`~/Library/Logs/firsthand/`. Control:

```bash
launchctl kickstart -k gui/$(id -u)/com.firsthand.server    # restart the API (after a code change)
launchctl bootout gui/$(id -u)/com.firsthand.tunnel          # stop the tunnel (site goes offline)
```

Keep the Mac awake while plugged in (one-time, needs your password):
`sudo pmset -c sleep 0 disksleep 0`.

## Operate by hand

```bash
export PATH="$HOME/.local/node/bin:$PATH"                 # node + vercel CLI live here (no Homebrew)
~/bin/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate &   # prints the *.trycloudflare.com address
deploy/vercel/make-site.sh https://<that-address>          # rewrites vercel.json
cd deploy/vercel/site && vercel deploy --prod --yes --scope asha22   # ~10 s; the scope is required
```

The quick tunnel's address changes every time it restarts, so those three
lines run again after a reboot. A named tunnel bound to your own domain
(`cloudflared tunnel login`, then `create`, `route dns`, `run`) has a fixed
address and needs the redeploy only once.

Start the server with `HUMANIZER_PUBLIC_URL=https://firsthand-navy.vercel.app`
(or the custom domain) so Google and Apple sign-in build their redirect URIs
with the public host rather than the tunnel host.

## Verified through Vercel (2026-09-13)

Landing, sign-in page, legal pages and stylesheet 200; sign-up 201 with the
session cookie set HttpOnly, SameSite=Lax, Secure; the app page 200 with the
cookie; engine health available on the 7B base. Streaming: a one-paragraph rewrite through the proxy delivered its first
progress frame after 0.2 s and finished at 41 s with a human verdict, so
Vercel passes Server-Sent Events through without buffering.

## Custom domain

Buy in the Vercel dashboard (Domains) or anywhere else, then
`vercel domains add <domain> firsthand` and follow the DNS instructions it
prints. Vercel issues the certificate. Then update `HUMANIZER_PUBLIC_URL` and
the OAuth redirect URIs to the new host.
