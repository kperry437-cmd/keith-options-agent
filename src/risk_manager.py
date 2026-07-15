"""Risk management for the options agent.

Enforces four hard limits derived from AgentConfig:
  - max_allocation_pct  : caps position size relative to portfolio equity
  - max_loss_pct        : triggers stop-loss when unrealised loss exceeds threshold
  - take_profit_pct     : triggers take-profit when unrealised gain exceeds threshold
  - max_hold_minutes    : forces exit when position age exceeds the time limit
"""
from __future__ import annotations

from datetime import datetime, timezone

from .config import AgentConfig


class RiskManager:
    """Evaluates whether a position should be opened or closed based on risk rules."""

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def max_position_value(self, portfolio_equity: float) -> float:
        """Return the maximum dollar value that may be committed to one position.

        Args:
            portfolio_equity: Current total equity of the portfolio in dollars.

        Returns:
            Dollar cap for the new position.
        """
        if portfolio_equity <= 0:
            raise ValueError("portfolio_equity must be positive")
        return portfolio_equity * self._cfg.max_allocation_pct

    def position_contracts(
        self,
        portfolio_equity: float,
        option_price: float,
        multiplier: int = 100,
    ) -> int:
        """Return the maximum number of contracts to buy for a new position.

        Args:
            portfolio_equity: Current total equity of the portfolio in dollars.
            option_price: Per-share premium of the option (e.g. $2.50).
            multiplier: Contract multiplier (standard equity options use 100).

        Returns:
            Number of whole contracts that fit within the allocation cap.
            Returns 0 when the option price exceeds the full cap.
        """
        if option_price <= 0:
            raise ValueError("option_price must be positive")
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        cap = self.max_position_value(portfolio_equity)
        return int(cap // (option_price * multiplier))

    # ------------------------------------------------------------------
    # Exit signal evaluation
    # ------------------------------------------------------------------

    def should_stop_loss(
        self,
        entry_cost: float,
        current_value: float,
        portfolio_equity: float,
    ) -> bool:
        """Return True when the position loss breaches max_loss_pct of portfolio equity.

        Args:
            entry_cost: Total cost basis of the position in dollars.
            current_value: Current market value of the position in dollars.
            portfolio_equity: Current portfolio equity used as the loss reference.

        Returns:
            True if the stop-loss threshold has been reached.
        """
        if portfolio_equity <= 0:
            raise ValueError("portfolio_equity must be positive")
        loss = entry_cost - current_value
        return loss >= portfolio_equity * self._cfg.max_loss_pct

    def should_take_profit(
        self,
        entry_cost: float,
        current_value: float,
    ) -> bool:
        """Return True when the position gain reaches take_profit_pct of entry cost.

        Args:
            entry_cost: Total cost basis of the position in dollars.
            current_value: Current market value of the position in dollars.

        Returns:
            True if the take-profit threshold has been reached.
        """
        if entry_cost <= 0:
            raise ValueError("entry_cost must be positive")
        gain_pct = (current_value - entry_cost) / entry_cost
        return gain_pct >= self._cfg.take_profit_pct

    def should_time_exit(self, opened_at: datetime) -> bool:
        """Return True when the position has been held longer than max_hold_minutes.

        Args:
            opened_at: Timezone-aware datetime when the position was opened.

        Returns:
            True if the position should be exited due to time.
        """
        now = datetime.now(timezone.utc)
        held_minutes = (now - opened_at).total_seconds() / 60
        return held_minutes >= self._cfg.max_hold_minutes

    def exit_signal(
        self,
        entry_cost: float,
        current_value: float,
        portfolio_equity: float,
        opened_at: datetime,
    ) -> str | None:
        """Evaluate all exit conditions and return the reason string, or None.

        Conditions are checked in priority order:
        1. stop_loss  – protects capital first
        2. take_profit – locks in gains
        3. time_exit  – max hold time exceeded

        Args:
            entry_cost: Total cost basis of the position in dollars.
            current_value: Current market value of the position in dollars.
            portfolio_equity: Current portfolio equity.
            opened_at: Timezone-aware datetime when the position was opened.

        Returns:
            One of ``"stop_loss"``, ``"take_profit"``, ``"time_exit"``, or ``None``.
        """
        if self.should_stop_loss(entry_cost, current_value, portfolio_equity):
            return "stop_loss"
        if self.should_take_profit(entry_cost, current_value):
            return "take_profit"
        if self.should_time_exit(opened_at):
            return "time_exit"
        return None
