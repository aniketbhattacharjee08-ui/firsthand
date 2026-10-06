# Taking Longhand public: runbook

This directory holds everything needed to run the existing app behind a login
and a Stripe paywall **without editing any of the existing code**. The paywall
lives in `src/humanizer/billing/` and wraps `humanizer.api.server.create_app`.

## What is gated and what is free

| Route | Cost |
|---|---|
| `GET /api/health`, `POST /api/analyze`, `POST /api/features`, references, `POST /api/plan` | free |
| `POST /api/humanize` (rule-based engine) | free, rate-limited |
| `POST /api/detect` | free, unless `detectors` names `gptzero` (spends the operator's key): 1 credit per 300 words |
| `POST /api/humanize/llm`, `/stream`, `/jobs` | sign-in + 1 credit per 300 words, times the effort multiplier |
| `POST /api/humanize/guided`, `/stream`, `/jobs` | sign-in + 1 credit per 300 words, times the effort multiplier |
| `GET /api/humanize/stream?job=` (SSE follow-up) | free; the job was charged when parked |
| `GET /api/models`, `POST /api/models/release` | operator only (`X-Admin-Token`) |

A request the server refuses (413 too long, 503 engine unavailable, any 4xx or
5xx) is refunded in full. Every response on a charged route carries
`X-Longhand-Credits-Charged` and `X-Longhand-Credits-Balance`.

Words are counted the way the server counts them (whitespace tokens). The
effort multiplier is `ceil(n_candidates / 6) x rounds x (1 + repair_attempts
// 3)`, so the defaults cost 1x and sixteen candidates over three rounds cost
9x; the 402 body reports it as `effort`. Set `LONGHAND_WORDS_PER_CREDIT` to
change the rate. Edit `RULES` in `billing/gate.py` to change what is gated.

## Local run

```bash
cp deploy/.env.example deploy/.env        # fill in LONGHAND_SECRET at least
set -a; source deploy/.env; set +a
.venv/bin/python -m humanizer.billing --check   # validates the environment
.venv/bin/python -m humanizer.billing --port 8000
```

With `LONGHAND_SMTP_URL` empty the sign-in link is printed to the server log.
With `LONGHAND_DEV_LINKS=1` it is also returned by `POST /api/auth/request-link`
as `dev_link`, which is convenient while building the frontend and must be
off in production.

`LONGHAND_PAYWALL=0` runs the same app with the auth and billing routes
mounted but nothing gated, so the frontend can be developed against the real
endpoints before anything is charged.

## Frontend contract

The frontend does not exist yet for these routes. When it is built, the flow is:

1. On load, `GET /api/me`. `user` is null when signed out. `paywall` says
   whether sign-in is needed for the LLM engines.
2. Sign-in form posts `{email}` to `POST /api/auth/request-link`. Tell the
   user to check their email. The link in the email is
   `GET /api/auth/verify?token=...&next=/`, which renders a small page that
   POSTs the token (so mail scanners that prefetch links cannot consume it),
   sets the cookie and goes to `next`. An invalid or expired token on GET
   redirects to `/?login_error=bad_token`; a link used twice shows the
   error on that page.
3. A `401 {error: "login_required"}` from any engine route means show the
   sign-in form. A `402 {error: "insufficient_credits", needed, balance}`
   means show packs from `GET /api/billing/packs` and, on choice, post
   `{pack}` to `POST /api/billing/checkout` and navigate to the returned
   `url`. Stripe returns the browser to `/?purchase=success` or
   `/?purchase=cancelled`; re-fetch `/api/me` to show the new balance. The
   webhook credits the account, so allow a second or two.
4. `POST /api/auth/logout` clears the session.

The session cookie is HttpOnly and `SameSite=Lax`; `fetch` on the same origin
sends it automatically. Programmatic callers use
`Authorization: Bearer lh_...` from `POST /api/auth/keys`.

## Google and Apple sign-in

The sign-in page shows "Continue with Google" and "Continue with Apple". Each
works once its keys are in `.env`; until then the button sends the person back
with a plain "not set up on this server yet" message.

- **Google**: create an OAuth client of type Web application in the Google
  Cloud console, add `https://<host>/api/auth/oauth/google/callback` as an
  authorised redirect URI, and set `GOOGLE_CLIENT_ID` and
  `GOOGLE_CLIENT_SECRET`. Works on `http://localhost:8000` too for testing.
- **Apple**: needs a paid Apple Developer account. Create an App ID with the
  Sign in with Apple capability, a Services ID (this is `APPLE_CLIENT_ID`)
  with the return URL `https://<host>/api/auth/oauth/apple/callback` (Apple
  refuses http), and a Sign in with Apple key; download the `.p8` once. Set
  `APPLE_TEAM_ID`, `APPLE_KEY_ID`, `APPLE_PRIVATE_KEY` (path or PEM). Install
  the extra: `pip install -e '.[oauth]'` (cryptography, for the ES256 client
  secret). Apple hides the email behind a relay address when the person
  chooses "Hide my email"; the account is created under that relay address.

Both flows are the standard authorization-code flow (PKCE for Google,
`form_post` for Apple); `GET /api/auth/providers` tells the page which are on.

## Stripe setup

1. Create one Product per pack with a one-time Price. Copy each `price_...`
   id into `STRIPE_PRICES` as `name:price_id:credits:label`.
2. Add a webhook endpoint at `https://<host>/api/billing/webhook` for the
   events `checkout.session.completed` and
   `checkout.session.async_payment_succeeded`. Copy the signing secret into
   `STRIPE_WEBHOOK_SECRET`.
3. Test locally with the Stripe CLI:
   `stripe listen --forward-to localhost:8000/api/billing/webhook` prints a
   temporary `whsec_...`; put it in `.env` for the session.

Credits are granted from the Checkout Session's `metadata.credits`, which the
server sets when it creates the session, so a tampered client cannot change
the amount. Webhook event ids are stored, so a retried event credits once.

## Public in ten minutes: a Cloudflare tunnel from the Mac you already have

Before renting anything, the running server can be public over https with no
router changes. `cloudflared` is installed at `~/bin/cloudflared` (downloaded
from Cloudflare's GitHub releases; no Homebrew on this machine).

```bash
~/bin/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate   # random *.trycloudflare.com address, no account
```

For your own domain, sign in once and bind a named tunnel (the browser opens
for the login):

```bash
~/bin/cloudflared tunnel login
~/bin/cloudflared tunnel create vervly
~/bin/cloudflared tunnel route dns vervly app.<your-domain>
HUMANIZER_PUBLIC_URL=https://app.<your-domain> humanizer serve --port 8000 &
~/bin/cloudflared tunnel run --url http://127.0.0.1:8000 vervly
```

The session cookie turns Secure behind the tunnel because Cloudflare sends
`X-Forwarded-Proto: https`. Verified 2026-09-13 through a quick tunnel: landing,
sign-up, app, engine health and legal pages all answered, and the cookie was
set with HttpOnly, SameSite=Lax and Secure. Limits: the Mac must stay awake and
connected, and there is no queue in front of the single GPU. Note this Mac's
own resolver sometimes returns NXDOMAIN for fresh tunnel hostnames while the
rest of the internet resolves them; test from a phone or with
`curl --resolve <host>:443:104.16.230.132`.

## Hosting

The MLX engines need Apple Silicon. Two shapes work:

- **Mac mini** (own hardware or a rental such as MacStadium or Scaleway).
  Install Caddy (`brew install caddy`), copy `deploy/Caddyfile` to
  `/opt/homebrew/etc/Caddyfile` with your hostname, `brew services start
  caddy`. Install the launchd job from `deploy/launchd/`. One machine handles
  one LLM rewrite at a time; the rule-based engine and detectors are cheap.
- **Linux VM** for a measurement-only launch: `deploy/systemd/longhand.service`.
  LLM routes return 503 and are refunded; the rule-based engine works.

Point DNS at the host; Caddy obtains the certificate. Set
`LONGHAND_TRUST_PROXY=1` so rate limits key on the real client address.

Back up the SQLite file (`LONGHAND_DB`). It holds users, credits and the
ledger. `sqlite3 billing.db ".backup backup.db"` is safe while the server
runs (WAL mode).

## Operator tasks

```bash
python -m humanizer.billing check                          # validate .env
python -m humanizer.billing admin user you@example.com     # account + ledger
python -m humanizer.billing admin grant you@example.com 20 --note promo
python -m humanizer.billing admin set-admin you@example.com
python -m humanizer.billing admin delete someone@example.com
python -m humanizer.billing admin stats --days 30
deploy/backup.sh                                          # also nightly via launchd
```

The request log (`LONGHAND_LOG_DIR/requests.log`) has one JSON line per
charged request: user id, path, words, credits, refunded, status, seconds,
and `unchanged` when the rewrite returned the draft. It never contains text.

## Concurrency and refunds

One GPU rewrite runs at a time (`LONGHAND_MAX_INFLIGHT`). A request that
cannot get the slot within `LONGHAND_QUEUE_WAIT_S` gets
`503 {error: "busy", retry_after_s}` and is refunded; the frontend should
retry once after `Retry-After`. Parked streaming jobs are charged when parked
and refunded if nobody connects within `LONGHAND_JOB_TTL_S`. A rewrite that
returns the whole document unchanged is refunded after the response
(`LONGHAND_REFUND_UNCHANGED`, at most `LONGHAND_UNCHANGED_REFUNDS_PER_DAY`
times per user, since a no-op run still occupies the GPU); the balance header
on that response still shows the charge, so re-fetch `/api/me` to display the
refund. A streaming run that ends with `event: error` and no `result` is
refunded too. At most `LONGHAND_MAX_WAITING` requests wait for the slot; the
rest are told busy at once.

## Frontend module

`deploy/frontend/` holds a drop-in `account.js` and `account.css` (sign-in
sheet, credits pill, packs sheet, account panel) that wrap `fetch` so
`app.js` needs no edits. `demo.html` fakes the API so every state can be
clicked through from disk. See `deploy/frontend/README.md`.

## Public in ten minutes: a Cloudflare tunnel from the Mac you already have

Before renting anything, the running server can be public over https with no
router changes. `cloudflared` is installed at `~/bin/cloudflared` (downloaded
from Cloudflare's GitHub releases; no Homebrew on this machine).

```bash
~/bin/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate   # random *.trycloudflare.com address, no account
```

For your own domain, sign in once and bind a named tunnel (the browser opens
for the login):

```bash
~/bin/cloudflared tunnel login
~/bin/cloudflared tunnel create vervly
~/bin/cloudflared tunnel route dns vervly app.<your-domain>
HUMANIZER_PUBLIC_URL=https://app.<your-domain> humanizer serve --port 8000 &
~/bin/cloudflared tunnel run --url http://127.0.0.1:8000 vervly
```

The session cookie turns Secure behind the tunnel because Cloudflare sends
`X-Forwarded-Proto: https`. Verified 2026-09-13 through a quick tunnel: landing,
sign-up, app, engine health and legal pages all answered, and the cookie was
set with HttpOnly, SameSite=Lax and Secure. Limits: the Mac must stay awake and
connected, and there is no queue in front of the single GPU. Note this Mac's
own resolver sometimes returns NXDOMAIN for fresh tunnel hostnames while the
rest of the internet resolves them; test from a phone or with
`curl --resolve <host>:443:104.16.230.132`.

## Hosting the page on Vercel

The Asha site's Vercel flow (push `main`, Vercel deploys a static directory)
works for `web/` too, but not for the API, which needs the Mac. See
`deploy/vercel/README.md` for the split and what to verify first.

## Before charging real money

- Switch the default base model to an Apache-2.0 checkpoint
  (`HUMANIZER_BASE_MODEL=mlx-community/Qwen2.5-7B-4bit`); Qwen2.5-3B is
  under the non-commercial Qwen Research licence.
- Check the licence on the surrogate and desklib detector model cards.
- Publish `deploy/legal/terms.md` and `deploy/legal/privacy.md` from the
  frontend (they are drafts; have them reviewed).
- Turn `LONGHAND_DEV_LINKS` off and confirm `LONGHAND_SMTP_URL` delivers.

## Merging into `humanizer serve`

When the frontend is ready, the whole merge is: make `cmd_serve` in
`src/humanizer/cli.py` call `humanizer.billing.app.run` instead of
`humanizer.api.run`, or change `server.run()` to build the app through
`humanizer.billing.app.create_app`. Nothing else moves.
