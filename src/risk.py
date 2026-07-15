"""
risk.py – Strict risk management checks and position-sizing helpers.

Risk rules enforced:
1.  Trading-hours gate: ignore runs outside the allowed window.
2.  Daily-loss limit: halt trading when cumulative P&L breaches the threshold.
3.  Max-open-positions cap: never exceed the configured limit.
4.  Duplicate-position guard: one position per symbol+direction at a time.
5.  Fractional position sizing: cap spend at max_position_size_pct of portfolio.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Tuple

import pytz

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")
_MARKET_OPEN_H = 9
_MARKET_OPEN_M = 30
_MARKET_CLOSE_H = 16
_MARKET_CLOSE_M = 0


# ---------------------------------------------------------------------------
# Individual checks (return bool)
# ---------------------------------------------------------------------------


def is_market_hours(
    now: datetime,
    open_buffer_min: int = 30,
    close_buffer_min: int = 30,
) -> bool:
    """Return True when *now* falls within the allowed trading window.

    The window is NYSE regular hours (9:30 – 16:00 ET) minus buffer periods
    at open and close to avoid elevated volatility.
    """
    now_et = now.astimezone(_ET)

    market_open = now_et.replace(
        hour=_MARKET_OPEN_H, minute=_MARKET_OPEN_M, second=0, microsecond=0
    )
    market_close = now_et.replace(
        hour=_MARKET_CLOSE_H, minute=_MARKET_CLOSE_M, second=0, microsecond=0
    )

    trade_start = market_open + timedelta(minutes=open_buffer_min)
    trade_end = market_close - timedelta(minutes=close_buffer_min)

    in_window = trade_start <= now_et <= trade_end
    if not in_window:
        logger.debug("Outside trading window (ET %s)", now_et.strftime("%H:%M"))
    return in_window


def check_daily_loss(
    daily_pnl: float,
    portfolio_value: float,
    max_daily_loss_pct: float,
) -> bool:
    """Return True when the daily loss limit has *not* been breached."""
    if portfolio_value <= 0:
        return False
    if daily_pnl >= 0:
        return True
    loss_pct = -daily_pnl / portfolio_value
    if loss_pct >= max_daily_loss_pct:
        logger.warning(
            "Daily loss limit reached: %.2f%% >= %.2f%%",
            loss_pct * 100,
            max_daily_loss_pct * 100,
        )
        return False
    return True


def check_position_count(open_positions: dict, max_open_positions: int) -> bool:
    """Return True when the open-position cap has *not* been reached."""
    count = len(open_positions)
    if count >= max_open_positions:
        logger.warning("Max open positions reached: %d/%d", count, max_open_positions)
        return False
    return True


def check_duplicate_position(
    open_positions: dict, symbol: str, signal_type: str
) -> bool:
    """Return True when no existing position exists for *symbol* + *signal_type*."""
    key = _position_key(symbol, signal_type)
    if key in open_positions:
        logger.info("Duplicate position blocked: %s", key)
        return False
    return True


# ---------------------------------------------------------------------------
# Composite gate
# ---------------------------------------------------------------------------


def passes_all_checks(
    state: dict,
    portfolio_value: float,
    symbol: str,
    signal_type: str,
    risk_cfg: dict,
    schedule_cfg: dict,
    now: datetime | None = None,
) -> Tuple[bool, str]:
    """Run every risk gate.

    Returns:
        (True, "OK") if all checks pass, or (False, reason) on first failure.
    """
    if now is None:
        now = datetime.now(pytz.UTC)

    if not is_market_hours(
        now,
        schedule_cfg.get("market_open_buffer_min", 30),
        schedule_cfg.get("market_close_buffer_min", 30),
    ):
        return False, "Outside trading hours"

    if not check_daily_loss(
        state.get("daily_pnl", 0.0),
        portfolio_value,
        risk_cfg["max_daily_loss_pct"],
    ):
        return False, "Daily loss limit reached"

    if not check_position_count(
        state.get("open_positions", {}),
        risk_cfg["max_open_positions"],
    ):
        return False, "Max open positions reached"

    if not check_duplicate_position(
        state.get("open_positions", {}), symbol, signal_type
    ):
        return False, f"Already have {signal_type} position for {symbol}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def calculate_position_size(
    portfolio_value: float,
    max_position_size_pct: float,
    contract_price: float,
) -> int:
    """Return the number of contracts to purchase.

    Each options contract controls 100 shares, so:
        contracts = floor(max_spend / (contract_price * 100))

    Returns at least 1 if the premium is non-zero, otherwise 0.
    """
    if contract_price <= 0 or portfolio_value <= 0:
        return 0
    max_spend = portfolio_value * max_position_size_pct
    contracts = int(max_spend / (contract_price * 100))
    return max(1, contracts)


# ---------------------------------------------------------------------------
# Stop-loss / profit-target evaluation
# ---------------------------------------------------------------------------


def should_close_position(
    entry_price: float,
    current_price: float,
    stop_loss_pct: float,
    profit_target_pct: float,
) -> Tuple[bool, str]:
    """Return (True, reason) when a position should be closed, else (False, "")."""
    if entry_price <= 0:
        return False, ""

    change_pct = (current_price - entry_price) / entry_price

    if change_pct <= -stop_loss_pct:
        return True, f"Stop-loss hit ({change_pct:.1%})"
    if change_pct >= profit_target_pct:
        return True, f"Profit target hit ({change_pct:.1%})"
    return False, ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _position_key(symbol: str, signal_type: str) -> str:
    return f"{symbol}_{signal_type}"
