#!/bin/sh
# Provision an Apple Silicon Mac (mini) as the Longhand host. Idempotent.
#   sh deploy/provision-mac.sh
# Run as the service user (default "longhand") after `git clone` into $HOME.
# Afterwards: copy deploy/.env.example to deploy/.env and fill it in, run
# deploy/sync-weights.sh from the dev machine, then `python -m humanizer.billing check`.
set -eu

REPO=${REPO:-"$HOME/humanizer"}
PY=${PY:-python3.11}

echo "== Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
brew install python@3.11 caddy rclone sqlite git >/dev/null

echo "== Python environment"
cd "$REPO"
[ -d .venv ] || $PY -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e ".[api,detectors]" mlx-lm

echo "== Directories"
mkdir -p data/cache/models data/adapters data/reference "$HOME/Library/Logs"

echo "== Power and firewall"
# Never sleep; the service must answer at 4am.
sudo pmset -a sleep 0 disksleep 0 displaysleep 10 womp 1 autorestart 1 || true
# Application firewall on; Caddy is the only listener that needs inbound.
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on || true
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$(brew --prefix)/bin/caddy" || true
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$(brew --prefix)/bin/caddy" || true

echo "== Caddy"
CADDYFILE="$(brew --prefix)/etc/Caddyfile"
if ! grep -q longhand "$CADDYFILE" 2>/dev/null; then
  cp deploy/Caddyfile "$CADDYFILE"
  echo "edit $CADDYFILE: set your hostname"
fi
# Caddy must bind 80/443: run it as root via brew services with sudo.
sudo brew services start caddy || true

echo "== launchd jobs"
for job in com.longhand.api com.longhand.backup; do
  sed -e "s#/Users/longhand/humanizer#$REPO#g" -e "s#/Users/longhand#$HOME#g" -e "s#<string>longhand</string>#<string>$(whoami)</string>#" \
    "deploy/launchd/$job.plist" > "/tmp/$job.plist"
  sudo cp "/tmp/$job.plist" "/Library/LaunchDaemons/$job.plist"
  sudo chown root:wheel "/Library/LaunchDaemons/$job.plist"
  sudo launchctl bootout system "/Library/LaunchDaemons/$job.plist" 2>/dev/null || true
  sudo launchctl bootstrap system "/Library/LaunchDaemons/$job.plist"
done

echo "== Done. Next:"
echo "   cp deploy/.env.example deploy/.env  &&  edit it"
echo "   (dev machine) deploy/sync-weights.sh $(whoami)@$(hostname):$REPO"
echo "   .venv/bin/python -m humanizer.billing check"
echo "   .venv/bin/python deploy/warm.py"
echo "   sudo launchctl kickstart -k system/com.longhand.api"
echo "   tail -f ~/Library/Logs/longhand.log"
echo ""
echo "Verify early: the log must show the MLX model loading under launchd."
echo "If Metal is unavailable to a LaunchDaemon, move the plist to"
echo "~/Library/LaunchAgents and enable automatic login for this user."
