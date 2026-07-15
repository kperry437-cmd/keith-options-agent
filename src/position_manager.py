"""File-backed position and daily P&L state management."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from .risk_manager import OPTION_CONTRACT_MULTIPLIER

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "state.json",
)


@dataclass
class Position:
    symbol: str         # option symbol, e.g. SPY250718C00560000
    underlying: str     # underlying equity symbol, e.g. SPY
    side: str           # "call" or "put"
    contracts: int      # number of contracts
    entry_price: float  # option premium per share at fill
    entry_time: str     # ISO-8601 datetime string
    current_price: float
    realized_pnl: float = 0.0
    closed: bool = False


class PositionManager:
    """Manages open positions and daily P&L with JSON-file persistence.

    Positions are persisted to *state_path* after every mutation.
    The daily P&L counter resets automatically at the start of each calendar
    day and closed positions from previous days are purged.
    """

    def __init__(self, state_path: str = _DEFAULT_STATE_PATH) -> None:
        self._path = state_path
        raw = self._load_raw()
        self._positions: Dict[str, Position] = {
            k: Position(**v) for k, v in raw.get("positions", {}).items()
        }
        self._realized_daily_pnl: float = raw.get("daily_pnl", 0.0)
        self._last_reset_date: str = raw.get(
            "last_reset_date", str(date.today())
        )
        self._maybe_reset_daily()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_raw(self) -> dict:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load state from %s: %s", self._path, exc)
            return {}

    def save(self) -> None:
        """Persist current state to disk."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        data = {
            "positions": {k: asdict(v) for k, v in self._positions.items()},
            "daily_pnl": self._realized_daily_pnl,
            "last_reset_date": self._last_reset_date,
        }
        try:
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.error("Failed to save state to %s: %s", self._path, exc)

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self) -> None:
        today = str(date.today())
        if self._last_reset_date != today:
            self._realized_daily_pnl = 0.0
            self._last_reset_date = today
            # Remove closed positions from prior days
            self._positions = {
                k: v for k, v in self._positions.items() if not v.closed
            }
            self.save()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def open_position(self, position: Position) -> None:
        """Record a newly opened position."""
        self._positions[position.symbol] = position
        self.save()

    def update_price(self, symbol: str, current_price: float) -> None:
        """Update the mark price for an open position."""
        pos = self._positions.get(symbol)
        if pos is not None and not pos.closed:
            pos.current_price = current_price

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close position at *exit_price* and return the realized P&L."""
        pos = self._positions.get(symbol)
        if pos is None or pos.closed:
            return 0.0
        pnl = (exit_price - pos.entry_price) * pos.contracts * OPTION_CONTRACT_MULTIPLIER
        pos.realized_pnl = pnl
        pos.closed = True
        pos.current_price = exit_price
        self._realized_daily_pnl += pnl
        self.save()
        return pnl

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if not p.closed]

    @property
    def open_position_count(self) -> int:
        return len(self.open_positions)

    @property
    def daily_pnl(self) -> float:
        """Realized P&L plus unrealized P&L for all open positions."""
        unrealized = sum(
            (p.current_price - p.entry_price) * p.contracts * OPTION_CONTRACT_MULTIPLIER
            for p in self.open_positions
        )
        return self._realized_daily_pnl + unrealized

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)
