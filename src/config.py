"""
config.py – Load and validate agent configuration from config.json and environment.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent
CONFIG_PATH = _ROOT / "config.json"

_REQUIRED_ENV = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")


def load_config() -> dict:
    """Return merged configuration dict from config.json and environment variables.

    Raises:
        FileNotFoundError: if config.json is missing.
        EnvironmentError: if required API credentials are not set.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}")

    with open(CONFIG_PATH) as fh:
        cfg = json.load(fh)

    missing = [key for key in _REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your Alpaca credentials."
        )

    cfg["alpaca_api_key"] = os.environ["ALPACA_API_KEY"]
    cfg["alpaca_secret_key"] = os.environ["ALPACA_SECRET_KEY"]
    cfg["alpaca_base_url"] = os.environ.get(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )

    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    """Raise ValueError for obviously invalid configuration values."""
    risk = cfg.get("risk", {})

    def _between(key: str, lo: float, hi: float) -> None:
        val = risk.get(key)
        if val is None or not (lo <= val <= hi):
            raise ValueError(
                f"risk.{key} must be between {lo} and {hi}, got {val!r}"
            )

    _between("max_position_size_pct", 0.001, 1.0)
    _between("max_daily_loss_pct", 0.001, 1.0)
    _between("stop_loss_pct", 0.01, 1.0)
    _between("profit_target_pct", 0.01, 100.0)
    _between("target_delta", 0.01, 0.99)

    if risk.get("max_open_positions", 0) < 1:
        raise ValueError("risk.max_open_positions must be >= 1")

    if risk.get("min_dte", 0) < 1:
        raise ValueError("risk.min_dte must be >= 1")

    if risk.get("min_dte", 0) >= risk.get("max_dte", 0):
        raise ValueError("risk.min_dte must be less than risk.max_dte")

    cross_pairs = cfg.get("cross_pairs", [])
    sma_periods = set(cfg.get("sma_periods", []))
    for fast, slow in cross_pairs:
        if fast >= slow:
            raise ValueError(
                f"cross_pairs entry [{fast}, {slow}]: fast period must be less than slow period"
            )
        if fast not in sma_periods or slow not in sma_periods:
            raise ValueError(
                f"cross_pairs entry [{fast}, {slow}]: both periods must be in sma_periods"
            )
