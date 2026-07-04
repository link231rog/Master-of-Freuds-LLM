#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "▸ MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-default}"
echo "▸ DEVICE=${DEVICE:-auto}"
echo "▸ PORT=${PORT:-5000}"

exec python3 scripts/webui.py
