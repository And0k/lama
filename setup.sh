#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Creating virtual environment .venv (if missing)"
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

echo "[setup] Activating virtualenv and upgrading pip"
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
  echo "[setup] Installing Python requirements from requirements.txt"
  pip install -r requirements.txt
fi

echo "[setup] Ensuring pytest is available"
pip install pytest

echo "[setup] Attempt to ensure bubblewrap and socat system packages (no-op if not available)"
if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then
    apt-get update && apt-get install -y bubblewrap socat || true
  else
    echo "[setup] not root, skipping apt install of bubblewrap/socat (they should be in image)"
  fi
fi

echo "[setup] Done. Activate environment with: source .venv/bin/activate"
