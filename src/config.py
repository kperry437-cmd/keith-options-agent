"""Configuration loader for the options agent."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    """Risk and execution parameters for the options agent."""

    # Maximum fraction of portfolio value to allocate to a single position.
    max_allocation_pct: float

    # Maximum fraction of portfolio value that can be lost on a single position
    # before triggering a stop-loss exit.
    max_loss_pct: float

    # Minimum gain on a position (as a fraction of entry cost) that triggers a
    # take-profit exit.
    take_profit_pct: float

    # Maximum number of minutes to hold an open position before forcing an exit
    # regardless of profit/loss.
    max_hold_minutes: int

    def __post_init__(self) -> None:
        if not 0 < self.max_allocation_pct <= 1:
            raise ValueError("max_allocation_pct must be in (0, 1]")
        if not 0 < self.max_loss_pct <= 1:
            raise ValueError("max_loss_pct must be in (0, 1]")
        if not 0 < self.take_profit_pct:
            raise ValueError("take_profit_pct must be positive")
        if self.max_hold_minutes <= 0:
            raise ValueError("max_hold_minutes must be positive")

    @classmethod
    def from_file(cls, path: str | os.PathLike[str] | None = None) -> "AgentConfig":
        """Load configuration from a JSON file.

        If *path* is not provided the function looks for ``config.json`` in the
        repository root (two levels above this file).
        """
        if path is None:
            path = Path(__file__).resolve().parent.parent / "config.json"
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls(
            max_allocation_pct=float(data["max_allocation_pct"]),
            max_loss_pct=float(data["max_loss_pct"]),
            take_profit_pct=float(data["take_profit_pct"]),
            max_hold_minutes=int(data["max_hold_minutes"]),
        )
