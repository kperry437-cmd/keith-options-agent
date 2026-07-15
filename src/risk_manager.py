"""Strict risk management rules for the options agent."""
from __future__ import annotations

OPTION_CONTRACT_MULTIPLIER = 100  # standard 100-share multiplier per contract

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from .config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str


@dataclass
class PositionSizeResult:
    contracts: int
    max_loss: float


class RiskManager:
    """Enforces all risk rules for trade decisions.

    All monetary values are in USD.  ``daily_pnl`` is negative for losses.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._risk = config.risk
        self._market_open_h = config.market_open_hour
        self._market_open_m = config.market_open_minute
        self._market_close_h = config.market_close_hour
        self._market_close_m = config.market_close_minute
        self._pre_close_buffer = config.pre_close_buffer_minutes
        self._post_open_buffer = config.post_open_buffer_minutes

    # ------------------------------------------------------------------
    # Trading-window check
    # ------------------------------------------------------------------

    def is_trading_window(self, now: datetime) -> RiskCheckResult:
        """Return whether *now* falls within the permitted trading window.

        ``now`` must be timezone-aware and expressed in the local market
        timezone (America/New_York for US equities).
        """
        market_open = now.replace(
            hour=self._market_open_h,
            minute=self._market_open_m,
            second=0,
            microsecond=0,
        )
        market_close = now.replace(
            hour=self._market_close_h,
            minute=self._market_close_m,
            second=0,
            microsecond=0,
        )
        post_open_cutoff = market_open + timedelta(minutes=self._post_open_buffer)
        pre_close_cutoff = market_close - timedelta(minutes=self._pre_close_buffer)

        if now < market_open:
            return RiskCheckResult(False, "Before market open")
        if now >= market_close:
            return RiskCheckResult(False, "After market close")
        if now < post_open_cutoff:
            return RiskCheckResult(
                False,
                f"Within {self._post_open_buffer} minutes of market open",
            )
        if now >= pre_close_cutoff:
            return RiskCheckResult(
                False,
                f"Within {self._pre_close_buffer} minutes of market close",
            )
        return RiskCheckResult(True, "Within trading window")

    # ------------------------------------------------------------------
    # Portfolio-level checks
    # ------------------------------------------------------------------

    def check_daily_loss(
        self, daily_pnl: float, portfolio_value: float
    ) -> RiskCheckResult:
        """Block trading when daily loss exceeds the configured threshold."""
        if portfolio_value <= 0:
            return RiskCheckResult(False, "Invalid portfolio value")
        loss_pct = -daily_pnl / portfolio_value
        if loss_pct >= self._risk.max_daily_loss_pct:
            return RiskCheckResult(
                False,
                f"Daily loss {loss_pct:.1%} exceeds limit "
                f"{self._risk.max_daily_loss_pct:.1%}",
            )
        return RiskCheckResult(True, "Daily loss within limit")

    def check_open_positions(self, open_position_count: int) -> RiskCheckResult:
        """Block new trades when the maximum open-position count is reached."""
        if open_position_count >= self._risk.max_open_positions:
            return RiskCheckResult(
                False,
                f"Max open positions ({self._risk.max_open_positions}) reached",
            )
        return RiskCheckResult(True, "Open positions within limit")

    # ------------------------------------------------------------------
    # Per-option checks
    # ------------------------------------------------------------------

    def check_option_delta(self, delta: float) -> RiskCheckResult:
        """Validate that |delta| falls within [min_delta, max_delta]."""
        abs_delta = abs(delta)
        if abs_delta < self._risk.min_delta:
            return RiskCheckResult(
                False,
                f"Delta {abs_delta:.2f} below minimum {self._risk.min_delta:.2f}",
            )
        if abs_delta > self._risk.max_delta:
            return RiskCheckResult(
                False,
                f"Delta {abs_delta:.2f} above maximum {self._risk.max_delta:.2f}",
            )
        return RiskCheckResult(True, "Delta within range")

    def check_dte(self, dte: int) -> RiskCheckResult:
        """Validate that the option DTE falls within [min_dte, max_dte]."""
        if dte < self._risk.min_dte:
            return RiskCheckResult(
                False,
                f"DTE {dte} below minimum {self._risk.min_dte}",
            )
        if dte > self._risk.max_dte:
            return RiskCheckResult(
                False,
                f"DTE {dte} above maximum {self._risk.max_dte}",
            )
        return RiskCheckResult(True, "DTE within range")

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        portfolio_value: float,
        option_price: float,
    ) -> PositionSizeResult:
        """Calculate the number of contracts within the per-trade risk budget.

        Limits total option premium to ``max_position_pct`` of
        ``portfolio_value``.  Each standard option contract covers 100 shares.
        Returns at least 1 contract when financially feasible, otherwise 0.
        """
        max_spend = portfolio_value * self._risk.max_position_pct
        cost_per_contract = option_price * OPTION_CONTRACT_MULTIPLIER
        if cost_per_contract <= 0:
            return PositionSizeResult(contracts=0, max_loss=0.0)
        contracts = int(max_spend / cost_per_contract)
        if contracts < 1:
            return PositionSizeResult(contracts=0, max_loss=0.0)
        max_loss = contracts * cost_per_contract * self._risk.option_stop_loss_pct
        return PositionSizeResult(contracts=contracts, max_loss=max_loss)

    # ------------------------------------------------------------------
    # Stop-loss check
    # ------------------------------------------------------------------

    def should_stop_loss(self, entry_price: float, current_price: float) -> bool:
        """Return True when the position loss meets or exceeds the stop threshold."""
        if entry_price <= 0:
            return False
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self._risk.option_stop_loss_pct

    # ------------------------------------------------------------------
    # Combined pre-trade approval
    # ------------------------------------------------------------------

    def approve_trade(
        self,
        *,
        now: datetime,
        daily_pnl: float,
        portfolio_value: float,
        open_position_count: int,
        option_delta: float,
        dte: int,
    ) -> RiskCheckResult:
        """Run every pre-trade risk check and return the first failure or approval."""
        checks: List[RiskCheckResult] = [
            self.is_trading_window(now),
            self.check_daily_loss(daily_pnl, portfolio_value),
            self.check_open_positions(open_position_count),
            self.check_option_delta(option_delta),
            self.check_dte(dte),
        ]
        for check in checks:
            if not check.allowed:
                logger.warning("Trade rejected: %s", check.reason)
                return check
        return RiskCheckResult(True, "All risk checks passed")
