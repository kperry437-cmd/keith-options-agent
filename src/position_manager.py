"""Position tracking for open options trades."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterator, Optional


@dataclass
class Position:
    """Represents a single open options position."""

    symbol: str
    option_type: str          # "call" or "put"
    strike: float
    expiration: str           # ISO date string e.g. "2026-07-18"
    contracts: int
    entry_price: float        # per-share premium paid
    multiplier: int = 100
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def entry_cost(self) -> float:
        """Total cost basis in dollars."""
        return self.entry_price * self.contracts * self.multiplier

    def current_value(self, current_price: float) -> float:
        """Current market value in dollars given a per-share premium."""
        return current_price * self.contracts * self.multiplier

    def unrealised_pnl(self, current_price: float) -> float:
        """Unrealised profit/loss in dollars."""
        return self.current_value(current_price) - self.entry_cost

    def unrealised_pnl_pct(self, current_price: float) -> float:
        """Unrealised profit/loss as a fraction of entry cost."""
        return self.unrealised_pnl(current_price) / self.entry_cost


class PositionManager:
    """Tracks all open positions and provides lookup/iteration helpers."""

    def __init__(self) -> None:
        # Key: a unique position id (e.g. "AAPL_call_210_2026-07-18")
        self._positions: Dict[str, Position] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_position_id(position: Position) -> str:
        return (
            f"{position.symbol}_{position.option_type}"
            f"_{position.strike}_{position.expiration}"
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, position: Position) -> str:
        """Record a new position and return its id."""
        pid = self.make_position_id(position)
        if pid in self._positions:
            raise ValueError(f"Position {pid!r} is already open")
        self._positions[pid] = position
        return pid

    def remove(self, position_id: str) -> Optional[Position]:
        """Remove and return a position by id, or None if not found."""
        return self._positions.pop(position_id, None)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, position_id: str) -> Optional[Position]:
        return self._positions.get(position_id)

    def all(self) -> Iterator[tuple[str, Position]]:
        yield from self._positions.items()

    def __len__(self) -> int:
        return len(self._positions)

    def __contains__(self, position_id: str) -> bool:
        return position_id in self._positions

    def summary(self, prices: Dict[str, float]) -> list[dict]:
        """Return a list of dicts with current P&L for each open position.

        Args:
            prices: Mapping of position_id -> current per-share option premium.
        """
        rows = []
        for pid, pos in self._positions.items():
            price = prices.get(pid)
            row: dict = {
                "position_id": pid,
                "symbol": pos.symbol,
                "option_type": pos.option_type,
                "strike": pos.strike,
                "expiration": pos.expiration,
                "contracts": pos.contracts,
                "entry_cost": pos.entry_cost,
                "opened_at": pos.opened_at.isoformat(),
            }
            if price is not None:
                row["current_value"] = pos.current_value(price)
                row["unrealised_pnl"] = pos.unrealised_pnl(price)
                row["unrealised_pnl_pct"] = pos.unrealised_pnl_pct(price)
            rows.append(row)
        return rows
