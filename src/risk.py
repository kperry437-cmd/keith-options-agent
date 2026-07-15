"""
risk.py – Position sizing, stop-loss, take-profit, and max-hold rules.

Rules (all configurable via ``config.json``):
  • Size  : 15 % of account equity per trade.
  • Stop  :  5 % below entry (option premium).
  • Target: 25 % above entry (option premium).
  • Max hold: 1 calendar day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskParams:
    """Configurable risk parameters."""

    size_pct: float = 0.15      # fraction of account equity per trade
    stop_pct: float = 0.05      # stop-loss as fraction of entry premium
    target_pct: float = 0.25    # take-profit as fraction of entry premium
    max_hold_days: int = 1      # maximum calendar days in a position


@dataclass
class PositionRisk:
    """Risk levels computed for one open position."""

    entry_premium: float        # cost basis per contract (in dollars)
    stop_price: float           # exit if premium falls to this
    target_price: float         # exit if premium rises to this
    max_exit_time: datetime     # exit no later than this UTC timestamp
    contracts: int              # number of contracts
    notional: float             # total cost: entry_premium * contracts * 100


def compute_risk(
    entry_premium: float,
    account_equity: float,
    params: Optional[RiskParams] = None,
    entry_time: Optional[datetime] = None,
) -> PositionRisk:
    """Compute stop, target, size, and expiry for a new position.

    Parameters
    ----------
    entry_premium:
        Mid-price of the option at entry (per-share, i.e. 1/100 of
        contract value for standard 100-multiplier contracts).
    account_equity:
        Current total account value in dollars.
    params:
        Risk parameters; defaults to ``RiskParams()``.
    entry_time:
        UTC datetime of entry; defaults to ``datetime.now(utc)``.

    Returns
    -------
    PositionRisk
    """
    if params is None:
        params = RiskParams()
    if entry_time is None:
        entry_time = datetime.now(tz=timezone.utc)
    if entry_premium <= 0:
        raise ValueError(f"entry_premium must be positive, got {entry_premium}")
    if account_equity <= 0:
        raise ValueError(f"account_equity must be positive, got {account_equity}")

    stop_price = entry_premium * (1.0 - params.stop_pct)
    target_price = entry_premium * (1.0 + params.target_pct)

    # Budget = size_pct * equity; each contract = premium * 100 (multiplier).
    budget = account_equity * params.size_pct
    contract_cost = entry_premium * 100.0
    contracts = max(1, int(budget / contract_cost))
    notional = entry_premium * contracts * 100.0

    max_exit_time = entry_time + timedelta(days=params.max_hold_days)

    risk = PositionRisk(
        entry_premium=entry_premium,
        stop_price=stop_price,
        target_price=target_price,
        max_exit_time=max_exit_time,
        contracts=contracts,
        notional=notional,
    )
    logger.info(
        "Risk computed: entry=%.4f stop=%.4f target=%.4f contracts=%d "
        "notional=%.2f max_exit=%s",
        entry_premium,
        stop_price,
        target_price,
        contracts,
        notional,
        max_exit_time.isoformat(),
    )
    return risk


def should_exit(
    current_premium: float,
    risk: PositionRisk,
    now: Optional[datetime] = None,
) -> tuple[bool, str]:
    """Determine whether an open position should be closed.

    Parameters
    ----------
    current_premium:
        Latest mid-price of the option.
    risk:
        The :class:`PositionRisk` computed at entry.
    now:
        Current UTC time; defaults to ``datetime.now(utc)``.

    Returns
    -------
    (should_exit, reason)
        A boolean flag and a human-readable reason string.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    if current_premium <= risk.stop_price:
        return True, (
            f"stop-loss hit (current={current_premium:.4f} "
            f"<= stop={risk.stop_price:.4f})"
        )
    if current_premium >= risk.target_price:
        return True, (
            f"take-profit hit (current={current_premium:.4f} "
            f">= target={risk.target_price:.4f})"
        )
    if now >= risk.max_exit_time:
        return True, (
            f"max-hold expired (now={now.isoformat()} "
            f">= max_exit={risk.max_exit_time.isoformat()})"
        )
    return False, "hold"
