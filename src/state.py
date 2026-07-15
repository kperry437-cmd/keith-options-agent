"""
state.py – Persist and restore agent state to / from a JSON file.

The state file stores the current open position (if any), the last signal
seen, and a run counter.  It is read at startup and written after every
cycle so that the agent survives process restarts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "state.json"


@dataclass
class PositionState:
    """Snapshot of the current open position."""

    symbol: str
    expiry: str
    strike: float
    option_type: str          # "call" or "put"
    entry_premium: float
    contracts: int
    notional: float
    stop_price: float
    target_price: float
    max_exit_time: str        # ISO-8601 UTC string
    entry_time: str           # ISO-8601 UTC string


@dataclass
class AgentState:
    """Top-level state object persisted to disk."""

    last_run: str = ""                        # ISO-8601 UTC
    run_count: int = 0
    last_signal: str = "none"
    position: Optional[PositionState] = None


def _default_encoder(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def load_state(path: str = DEFAULT_STATE_FILE) -> AgentState:
    """Load :class:`AgentState` from *path*.

    Returns a fresh :class:`AgentState` if the file does not exist or is
    corrupt.
    """
    p = Path(path)
    if not p.exists():
        logger.info("State file %s not found; starting with empty state", path)
        return AgentState()

    try:
        with p.open("r", encoding="utf-8") as fh:
            data: dict = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read state file %s: %s; resetting state", path, exc)
        return AgentState()

    position_data = data.pop("position", None)
    position = PositionState(**position_data) if position_data else None

    state = AgentState(**data, position=position)
    logger.debug("Loaded state: run_count=%d last_signal=%s", state.run_count, state.last_signal)
    return state


def save_state(state: AgentState, path: str = DEFAULT_STATE_FILE) -> None:
    """Serialise *state* and write it atomically to *path*.

    Uses a write-then-rename pattern to avoid partial writes.
    """
    p = Path(path)
    tmp = Path(str(p) + ".tmp")

    payload: dict = {
        "last_run": state.last_run,
        "run_count": state.run_count,
        "last_signal": state.last_signal,
        "position": asdict(state.position) if state.position else None,
    }

    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_default_encoder)
        tmp.replace(p)
        logger.debug("State saved to %s", path)
    except OSError as exc:
        logger.error("Failed to save state to %s: %s", path, exc)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def open_position_from_state(state: AgentState) -> Optional[PositionState]:
    """Return the current open position, or ``None``."""
    return state.position


def clear_position(state: AgentState) -> AgentState:
    """Return a copy of *state* with ``position`` set to ``None``."""
    return AgentState(
        last_run=state.last_run,
        run_count=state.run_count,
        last_signal=state.last_signal,
        position=None,
    )
