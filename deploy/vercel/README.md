# Option B: Vercel for the page, the Mac for the API

The Asha Berkeley site is a static site on Vercel: push to `main`, Vercel
deploys the `asha-site/` directory. Longhand's `web/` is also three static
files, so the same flow works for the page. What cannot move to Vercel is
the API: it needs a Mac with Apple Silicon for the MLX engines and several
gigabytes of resident model weights, which no serverless platform offers.

So the split is:

- `longhand.example.com` on Vercel, root directory `web/`, this `vercel.json`
  at the repo root (or set Vercel's root directory to `web/` and put the
  file there).
- `api.longhand.example.com` on the Mac behind Caddy, from `deploy/Caddyfile`
  with that hostname.
- The rewrite above proxies `/api/*` through Vercel to the Mac, so the
  browser sees one origin and the session cookie works unchanged.

Two things to verify in staging before choosing this over option A (Caddy
serves everything from the Mac, the default in `deploy/README.md`):

1. **Server-Sent Events through the rewrite.** Vercel proxies external
   rewrites as a stream, but confirm frames arrive incrementally with
   `curl -N https://longhand.example.com/api/humanize/stream ...`. If they
   buffer, keep the page on Vercel and point only the streaming call at
   `api.longhand.example.com` directly (this needs the CORS origin regex in
   `server.py` extended to the Vercel host and `SameSite=None` cookies; a
   merge-day edit).
2. **Long blocking requests.** A 60 s `POST /api/humanize/llm` must not be
   cut by the proxy. Test one 1,000-word document.

If both pass, option B gives you the Vercel workflow you already use, preview
deployments on the `dev` branch, and a CDN in front of the page. The Mac
still needs everything in `deploy/README.md` except serving `web/`: set
`HUMANIZER_WEB_DIR` to an empty directory so the API host serves no page,
and point `LONGHAND_PUBLIC_URL` at the Vercel hostname so login links and
Stripe return URLs land on the page.
