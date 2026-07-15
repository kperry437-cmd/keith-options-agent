"""
state.py – Persist agent state across 5-minute cron runs.

State is stored as JSON on disk at ``<project_root>/state.json``.
The daily_pnl counter resets automatically on the first run of each calendar day.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytz

logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).parent.parent / "state.json"
_UTC = pytz.UTC

_DEFAULT_STATE: dict = {
    "daily_pnl": 0.0,
    "open_positions": {},
    "last_reset_date": None,
}


def load_state(path: Path | None = None) -> dict:
    """Load state from disk, returning defaults if the file does not exist."""
    state_path = path or _STATE_FILE
    if state_path.exists():
        try:
            with open(state_path) as fh:
                loaded = json.load(fh)
            # Merge with defaults so new keys are always present
            return {**_DEFAULT_STATE, **loaded}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read state file (%s), using defaults", exc)
    return dict(_DEFAULT_STATE)


def save_state(state: dict, path: Path | None = None) -> None:
    """Persist *state* to disk atomically."""
    state_path = path or _STATE_FILE
    tmp_path = state_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as fh:
        json.dump(state, fh, indent=2, default=str)
    tmp_path.replace(state_path)


def reset_daily_state_if_needed(state: dict, now: datetime | None = None) -> dict:
    """Zero out daily_pnl when the calendar date has advanced.

    This is idempotent – multiple calls on the same day are safe.
    """
    if now is None:
        now = datetime.now(_UTC)
    today = now.strftime("%Y-%m-%d")
    if state.get("last_reset_date") != today:
        logger.info("New trading day – resetting daily P&L")
        state["daily_pnl"] = 0.0
        state["last_reset_date"] = today
    return state


def record_fill(
    state: dict,
    position_key: str,
    contract_symbol: str,
    qty: int,
    fill_price: float,
    signal: dict,
    order_id: str,
    now: datetime | None = None,
) -> dict:
    """Add a newly filled position to state."""
    if now is None:
        now = datetime.now(_UTC)
    state["open_positions"][position_key] = {
        "order_id": order_id,
        "contract": contract_symbol,
        "qty": qty,
        "entry_price": fill_price,
        "signal": signal,
        "opened_at": now.isoformat(),
    }
    return state


def close_position(
    state: dict,
    position_key: str,
    close_price: float,
) -> dict:
    """Remove a position from state and update daily P&L."""
    pos = state["open_positions"].pop(position_key, None)
    if pos is None:
        logger.warning("Tried to close unknown position: %s", position_key)
        return state

    entry = pos.get("entry_price", 0.0)
    qty = pos.get("qty", 0)
    # Standard equity options: each contract controls 100 shares
    _SHARES_PER_CONTRACT = 100
    pnl = (close_price - entry) * qty * _SHARES_PER_CONTRACT
    state["daily_pnl"] = state.get("daily_pnl", 0.0) + pnl
    logger.info(
        "Closed %s: entry=%.2f close=%.2f qty=%d P&L=$%.2f",
        position_key,
        entry,
        close_price,
        qty,
        pnl,
    )
    return state
