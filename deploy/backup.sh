#!/bin/sh
# Nightly backup of the Longhand billing database.
#   deploy/backup.sh [dest_dir]
# Uses sqlite3's online .backup (safe while the server runs; the store is in
# WAL mode), keeps the newest 30 copies, and, when RCLONE_REMOTE is set
# (e.g. "b2:longhand-backups"), copies the new file off the machine.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
if [ -f "$HERE/.env" ]; then
  set -a; . "$HERE/.env"; set +a
fi
DB=${LONGHAND_DB:-"$HERE/../data/cache/billing.db"}
DEST=${1:-${LONGHAND_BACKUP_DIR:-"$HERE/../data/cache/backups"}}
mkdir -p "$DEST"
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT="$DEST/billing-$STAMP.db"

sqlite3 "$DB" ".backup '$OUT'"
sqlite3 "$OUT" "PRAGMA integrity_check;" | grep -q '^ok$'
gzip -f "$OUT"
echo "backup written: $OUT.gz"

# Keep the newest 30.
ls -1t "$DEST"/billing-*.db.gz 2>/dev/null | tail -n +31 | xargs -r rm -f

if [ -n "${RCLONE_REMOTE:-}" ] && command -v rclone >/dev/null 2>&1; then
  rclone copy "$OUT.gz" "$RCLONE_REMOTE/" && echo "copied to $RCLONE_REMOTE"
fi
