#!/usr/bin/env bash
# bootstrap.sh — Environment setup for keith-options-agent (Unix/macOS only)
#
# Usage:
#   ./bootstrap.sh
#
# This script creates a Python virtual environment (if one does not already
# exist), installs all project dependencies from requirements.txt, and copies
# .env.example to .env (if .env does not yet exist) so you can fill in your
# Alpaca API credentials before running the agent.
#
# NOTE: This script targets Unix-like systems (Linux, macOS). On Windows,
# use Git Bash or WSL, or activate the virtual environment manually via
# .venv\Scripts\activate.

set -euo pipefail

VENV_DIR=".venv"

echo "==> Checking Python version..."
python3 --version

# Create virtual environment only when it does not already exist.
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment in ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
else
    echo "==> Virtual environment already exists at ${VENV_DIR}, skipping creation."
fi

echo "==> Activating virtual environment..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "==> Installing dependencies from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt

# Copy the example environment file so the user only needs to fill in secrets.
if [ ! -f ".env" ]; then
    if [ ! -f ".env.example" ]; then
        echo "WARNING: .env.example not found — skipping .env creation. Create .env manually."
    else
        echo "==> Copying .env.example to .env — please fill in your API credentials."
        cp .env.example .env
    fi
else
    echo "==> .env already exists, skipping copy."
fi

echo ""
echo "Bootstrap complete. Activate the virtual environment with:"
echo "    source ${VENV_DIR}/bin/activate"
echo "Then edit .env with your Alpaca API credentials before running the agent."
