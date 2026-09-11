# Longhand launch plan

Written 2026-09-09, after the paywall package landed in `src/humanizer/billing/`
with no edits to existing code. This is the execution plan for making the
site public. It is split by *when* each piece can happen: now (backend only,
no conflict with the frontend rewrite), at merge (the day the frontend is
done), and operator steps that need accounts, money or DNS.

Facts this plan is built on, checked in the repo on the date above:

- The server is one uvicorn process. LLM generation is serialised per
  backend by an `RLock` in `llm.MlxBackend`; requests beyond the first wait in
  the threadpool with no queue limit and no "busy" answer.
- `POST /api/humanize/jobs` parks a request for `EventSource`; the job holds
  the body for 900 s and is single-use. The gate charges at park time.
- Weights the product needs that are **not** on the Hugging Face hub live in
  gitignored paths: the GPTZero surrogate (`data/cache/models/gptzero-surrogate`,
  492 MB), the register model (`data/cache/models/qwen3-4b-instruct-4bit`,
  2.1 GB) and the LoRA adapters (`data/adapters/`, 84 MB). Hub weights needed
  at runtime: `mlx-community/Qwen2.5-7B-4bit` (~4.5 GB) and, for research
  routes, desklib (1.7 GB). The dev machine's 50 GB hub cache is not needed.
- `GPTZeroClient` caches every response on disk under `data/cache/gptzero`,
  keyed by text hash, including calls made with a *caller's* key. The
  privacy policy must say so, or the cache must be disabled for caller keys.
- The frontend (`web/`) already sends `facts` and an optional
  `gptzero_api_key`. It has no sign-in, balance or purchase UI yet.
- Dev machine: Apple M4 Pro, 24 GB. Suitable as a staging box, not as the
  production host (residential connection, sleep, no static IP).

---

## 0. Decisions only the owner can make

Answer these before the merge day; everything below is parameterised on them.

| # | Decision | Default if unanswered |
|---|---|---|
| D1 | Domain name | needs purchase; plan assumes `longhand.<tld>` |
| D2 | Hosting: rented Mac mini (MacStadium / Scaleway / OakHost) vs own hardware in a colo | rented M4 mini, 24 GB |
| D3 | Pricing: words per credit, pack sizes, prices | 300 words = 1 credit; packs 50 / 250 / 1000 credits at $5 / $20 / $60 |
| D4 | Free tier: signup credits | 3 credits (about 900 words of LLM rewrite) |
| D5 | Transactional email provider (Postmark, Resend, SES) | Postmark SMTP |
| D6 | Legal entity and payout account for Stripe | sole proprietor, owner's bank |
| D7 | Refund stance in the terms | credits refunded on refused requests only; no cash refunds |
| D8 | Whether an entire-document-unchanged rewrite is refunded | yes (good will, cheap) |
| D9 | Base model licence: 7B (Apache, 4 of 9 on GPTZero), 1.5B (Apache, unmeasured), or a commercial licence for 3B (7 of 9) | 7B until 1.5B is benched |

---

## 1. Now: backend work that touches only `humanizer.billing`

**Status 2026-09-10: 1.1 through 1.10 are implemented and tested**
(`tests/test_billing.py`, 44 tests). Scripts are in `deploy/`. An
adversarial security review of the package was run the same day and its
findings fixed: duplicate `job` query values (a free-rewrite exploit),
`event: error` inside a 200 stream not refunded, ledger transactions that
could wedge the connection, pricing that disagreed with the server's word
count, unknown jobs taking a GPU slot, effort (candidates, rounds) not
priced, webhook dedupe recorded before the credit, leftmost
X-Forwarded-For, GET verify consuming the nonce, dev links allowed on https,
no body limit on free POSTs, non-constant-time admin token compare. Known
and accepted: the signup cap reveals whether an email exists; chargebacks
(`charge.dispute.created`) are not handled and must be settled by hand with
`admin grant` of a negative amount. The frontend account module (Phase 2
reference implementation) is in `deploy/frontend/`.

Hosting note from the Asha Berkeley project: that site deploys as a static
directory on Vercel from `main`. Longhand's page can use the same flow, the
API cannot (Apple Silicon, resident weights). `deploy/vercel/` holds the
hybrid configuration and the two things to verify before choosing it.

**1.1 Concurrency guard for the GPU.** In `gate.py`, a semaphore around
charged routes: `LONGHAND_MAX_INFLIGHT` (default 1) with a bounded wait
`LONGHAND_QUEUE_WAIT_S` (default 20). Beyond the wait, answer
`503 {error: "busy", retry_after_s}` and refund. Without this, ten
simultaneous users each wait for all the others with no feedback and the
process can exhaust memory loading a second backend.

**1.2 Parked-job refunds.** The gate already charges `POST /api/humanize/jobs`.
Add: parse `job_id` from the response, remember `(job_id, user, cost, t)`;
when `GET /api/humanize/stream?job=` arrives, mark consumed; a sweep refunds
any job not consumed within 900 s (the server's TTL). Same for guided jobs.
Otherwise a closed tab between park and connect loses the credits.

**1.3 Refund when nothing changed (D8).** For the blocking route, read the
response body and refund if `summary.paragraphs_unchanged == paragraphs`
(field names to confirm in `pipeline.LlmHumanizeResult.as_dict`). For SSE,
watch for the `finalize/done` frame in the stream passing through the send
wrapper. Both are contained in `gate.py`.

**1.4 Account deletion.** `DELETE /api/me`: revoke sessions and keys, null the
email (keep ledger rows for accounting under a tombstone id). Required for a
credible privacy policy.

**1.5 Operator CLI.** `python -m humanizer.billing admin <cmd>`: `grant
<email> <n> [note]`, `set-admin <email>`, `user <email>`, `stats` (signups,
credits sold, credits spent, refunds, by day). Replaces the Python snippets
in `deploy/README.md`.

**1.6 Structured request log.** One JSON line per charged request to
`LONGHAND_LOG_DIR/requests.log`: time, user id, path, words, credits, status,
seconds. This is the data for pricing and for spotting abuse. No text is
logged.

**1.7 Free-credit abuse limits.** `LONGHAND_SIGNUPS_PER_IP_PER_DAY` (default
3) enforced in `request_link`; optional disposable-domain blocklist file
`LONGHAND_EMAIL_BLOCKLIST`.

**1.8 Backup script.** `deploy/backup.sh`: `sqlite3 .backup` to a dated file,
keep 30, optional `rclone copy` to an off-box bucket. Paired launchd job
`deploy/launchd/com.longhand.backup.plist` at 03:00.

**1.9 Provisioning script.** `deploy/provision-mac.sh`, idempotent: Homebrew,
Python 3.11 (the repo runs on 3.9 today; 3.11 is what the mini will have),
git clone, venv, `pip install -e '.[api,detectors]' mlx-lm`, Caddy, the
launchd jobs, `pmset -a sleep 0 disksleep 0`, pf rules allowing only 22, 80,
443. Plus `deploy/sync-weights.sh` to rsync the three gitignored weight
directories from the dev machine and `deploy/warm.py` to pre-download hub
checkpoints and run one rewrite so the first user does not wait for a
download.

**1.10 Tests** for 1.1 through 1.7 in `tests/test_billing.py`.

Estimated effort: two working days including tests.

---

## 2. Frontend contract (the owner's work in `web/`)

Already documented in `deploy/README.md`. Additions the plan needs:

- Header state from `GET /api/me`: signed out (Sign in), signed in (email,
  credits). A sign-in sheet posting to `/api/auth/request-link`, then a
  "check your email" state. Handle `/?login_error=` on load.
- A 401 from an engine route opens the sign-in sheet; a 402 opens a packs
  sheet built from `/api/billing/packs`, posting the chosen pack to
  `/api/billing/checkout` and navigating to `url`. On `/?purchase=success`
  poll `/api/me` for up to ten seconds for the balance to rise.
- Show the cost before running: `ceil(words / words_per_credit)` using
  `words_per_credit` from `/api/billing/health`, next to the Humanize button.
- Read `X-Longhand-Credits-Balance` from engine responses to update the
  header without another request.
- An account view: ledger (`/api/billing/ledger`), API keys, sign out,
  delete account.
- Footer links to `/legal/terms.html` and `/legal/privacy.html`. The pages
  are static files in `web/legal/` generated from `deploy/legal/*.md` at
  merge time (see 3.4).
- A 503 `busy` answer shows "the rewriter is busy, retrying in N s" and
  retries once.

---

## 3. Merge day: code changes to existing files

Exact snippets are in `deploy/merge-day.md`. Small and mechanical. Do them on a branch, run the full suite, then deploy.

**3.1 Entry point.** `src/humanizer/cli.py: cmd_serve` imports
`humanizer.billing.app.run` instead of `humanizer.api.run`. One import, one
call. `humanizer serve` then serves the paywalled app; `LONGHAND_PAYWALL`
unset keeps it open for local development.

**3.2 `pyproject.toml`.** Add `longhand = "humanizer.billing.__main__:main"`
under `[project.scripts]`. No new dependencies.

**3.3 Licences (D9).** `llm.FREEFORM_MODEL` defaults to
`mlx-community/Qwen2.5-3B-4bit`, Qwen Research licence, non-commercial. The
repo measured 3B at 7 of 9 and 7B at 4 of 9 on GPTZero (research/24 §6.5),
so the licence-safe swap costs pass rate. See `deploy/merge-day.md` §4 for
the three options. Also confirm the surrogate's base checkpoint licence and
desklib's.

**3.4 Legal pages.** Render `deploy/legal/*.md` to `web/legal/*.html` in the
site's own styles. Static, no build step, matching `web/DESIGN.md`.

**3.5 Security headers.** In `deploy/Caddyfile` add a Content-Security-Policy
that allows `self`, Google Fonts (`fonts.googleapis.com`,
`fonts.gstatic.com`) and `checkout.stripe.com` for the redirect. Test that
the page still loads its fonts.

**3.6 GPTZero cache and caller keys.** Either pass `cache_dir=None` to the
`GPTZeroClient` built in `server._request_judge` (a caller's text should
not be stored on the operator's disk) or state the retention in the privacy
policy. Recommend the former: two-line change.

**3.7 README.** Replace the Install section's `humanizer serve` note with the
paywall paragraph and a pointer to `deploy/`.

**3.8 Tests.** Full suite plus `tests/test_billing.py`; add one test that
`humanizer serve --help` still works and that `create_app` from
`humanizer.billing` serves `web/` when present.

Estimated effort: half a day.

---

## 4. Operator steps (accounts, DNS, hardware)

**4.1 Domain (D1).** Buy it. DNS: `A`/`AAAA` for the apex and `www` to the
host; `CAA` record for Let's Encrypt; SPF, DKIM and DMARC records from the
email provider (4.4) so sign-in links are not junked.

**4.2 Host (D2).** Rent an Apple Silicon mini with 24 GB and 256 GB. Run
`deploy/provision-mac.sh`, `deploy/sync-weights.sh`, `deploy/warm.py`.
Verify one thing early: Metal is reachable from a `LaunchDaemon` without a
logged-in user. If not, switch the plist to a `LaunchAgent` with automatic
login for the service user and FileVault off, or use the vendor's
auto-login option.

**4.3 Stripe (D3, D6).** Business profile and payout account. One Product per
pack, one-time Prices. Webhook endpoint
`https://<domain>/api/billing/webhook` for `checkout.session.completed` and
`checkout.session.async_payment_succeeded`. Stripe requires a public refund
policy and terms; 3.4 covers that. Enable Stripe Tax if selling to the EU/UK.
Do the whole flow in test mode first (4.7), then swap to live keys.

**4.4 Email (D5).** Provider account, verified sending domain, SMTP
credentials into `LONGHAND_SMTP_URL`. Send one link to a Gmail and one to an
Outlook address and check they arrive in the inbox.

**4.5 Monitoring.** External uptime check on `GET /api/health` every minute
(Better Stack or UptimeRobot, free tiers). `newsyslog` rotation for the two
launchd log files. A disk-space alert at 80 percent: model downloads and the
GPTZero cache grow.

**4.6 Backups.** Install `com.longhand.backup.plist`; confirm a restore by
opening the copy with `sqlite3` and counting users.

**4.7 Staging rehearsal on the dev M4 Pro.** `LONGHAND_PAYWALL=1`, Stripe
test keys, `stripe listen` forwarding the webhook, `LONGHAND_DEV_LINKS=1`.
Walk the whole path: sign in, run out of credits, buy a pack with card
`4242 4242 4242 4242`, see the balance rise, run a rewrite, confirm the
charge header, send a 1,300-word document and confirm the 413 is refunded,
open three tabs and confirm the busy answer, close a tab between park and
stream and confirm the refund after 15 minutes.

---

## 5. Launch checklist

In order. Each line is a yes/no.

1. `python -m humanizer.billing --check` on the host prints `paywall True`,
   `stripe True`, `mail smtp`.
2. `LONGHAND_DEV_LINKS` is unset. `LONGHAND_TRUST_PROXY=1`.
   `LONGHAND_PUBLIC_URL` is `https://…` so cookies are `Secure`.
3. `curl -I https://<domain>/` shows HSTS and the CSP; `/api/health` is 200.
4. Rate limit keys on the real client IP (two requests from two networks
   show separate limits in the request log).
5. SSE works through Caddy: `curl -N` on a stream shows frames arriving
   incrementally, not at the end.
6. Sign-in email arrives within a minute at two providers.
7. Live-mode purchase of the smallest pack from a real card; refund it in
   the Stripe dashboard afterwards; confirm the webhook event shows
   `credited` in the log.
8. Legal pages load and the footer links to them.
9. Backup job has produced one file and it opens.
10. Uptime monitor is green and alerts to a phone.
11. Default model is Apache-licensed (`/api/humanize/llm/health` shows
    `Qwen2.5-7B-4bit`).
12. Announce.

---

## 6. Capacity and pricing worksheet (D3)

Measured: a 6-candidate rewrite of a ~150-word paragraph takes ~12 s on an
M4 Pro with the 3B model; the 7B base is roughly half the speed. Budget
60 s per 500-word document. One mini therefore serves about 1,400 documents
a day at full utilisation, realistically 300 to 500 with idle time.

Fixed cost, rented mini plus domain, email and monitoring: roughly $120 to
$180 a month. At the default pricing (300 words = 1 credit, $5 for 50
credits) a 500-word document costs the user 20 cents and the operator about
1 cent of amortised compute at 300 documents a day. The GPTZero surrogate is
free; a user who supplies their own GPTZero key pays GPTZero directly. Break-
even is about 1,000 credits sold a month. Adjust `LONGHAND_WORDS_PER_CREDIT`
and `STRIPE_PRICES` without code changes.

If demand exceeds one machine: a second mini behind Caddy round-robin needs
the SQLite store replaced by Postgres or a shared Redis for sessions,
credits and the job table. That is the one architectural change in the
billing package (`store.py` behind the same interface) and is not needed for
launch.

---

## 7. Risks

- **Efficacy claim.** Best measured pass rate on live GPTZero is 7 of 9
  bench paragraphs. The site shows the real before/after verdict; the
  marketing must not promise a pass. The terms say so.
- **Metal from a daemon** (4.2). Verify on day one of the host.
- **Single process.** A crash mid-rewrite loses in-flight jobs; launchd
  restarts within seconds; the gate refunds on exception. Acceptable.
- **Free-credit farming.** Bounded by 1.7 and the email rate limit. Watch
  the request log the first week.
- **Chargebacks.** Credits are consumed on use; keep the ledger and the
  request log so a dispute can be answered with dates and word counts.
- **Model licences.** Covered by 3.3. Do not launch on the 3B checkpoint.
