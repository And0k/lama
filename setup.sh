#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Creating virtual environment .venv (if missing)"
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

echo "[setup] Activating virtualenv"
source .venv/bin/activate

echo "[setup] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
  echo "[setup] Installing requirements.txt"
  pip install -r requirements.txt
fi

echo "[setup] Ensuring pytest is available"
pip install pytest

echo "[setup] Done"