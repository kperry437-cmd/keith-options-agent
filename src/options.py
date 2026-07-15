"""
options.py – Select near-ATM option contracts with ≤ 5 DTE.

Uses yfinance option chains.  Given a signal direction, the module picks the
closest available expiry within *max_dte* calendar days, then selects the
strike nearest to the current spot price (at-the-money).

Call options are selected for bullish (golden-cross) signals;
put options are selected for bearish (death-cross) signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from src.signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Lightweight description of a selected option contract."""

    symbol: str
    expiry: str           # "YYYY-MM-DD"
    strike: float
    option_type: str      # "call" or "put"
    last_price: float
    bid: float
    ask: float
    implied_volatility: float
    open_interest: int
    dte: int              # calendar days to expiry


def _dte(expiry_str: str) -> int:
    """Return calendar days between today (UTC) and *expiry_str*."""
    today = datetime.now(tz=timezone.utc).date()
    expiry_date = date.fromisoformat(expiry_str)
    return (expiry_date - today).days


def select_option(
    symbol: str,
    signal: Signal,
    max_dte: int = 5,
    min_open_interest: int = 10,
) -> Optional[OptionContract]:
    """Return the best near-ATM contract for *symbol* given *signal*.

    Parameters
    ----------
    symbol:
        Underlying ticker, e.g. ``"SPY"``.
    signal:
        ``Signal.GOLDEN_CROSS`` → call; ``Signal.DEATH_CROSS`` → put.
        Returns ``None`` for ``Signal.NONE``.
    max_dte:
        Maximum calendar days to expiry (inclusive).
    min_open_interest:
        Minimum open interest required to accept a contract.

    Returns
    -------
    OptionContract or None
    """
    if signal == Signal.NONE:
        return None

    option_type = "call" if signal == Signal.GOLDEN_CROSS else "put"

    ticker = yf.Ticker(symbol)

    # Current spot price.
    info = ticker.fast_info
    spot = float(info.get("lastPrice") or info.get("last_price") or 0)
    if spot == 0:
        hist = ticker.history(period="1d", interval="1m")
        if hist.empty:
            raise ValueError(f"Cannot determine spot price for {symbol!r}")
        spot = float(hist["Close"].iloc[-1])

    logger.debug("Spot price for %s: %.4f", symbol, spot)

    # Filter expiries within max_dte.
    all_expiries = ticker.options  # tuple of "YYYY-MM-DD" strings
    valid_expiries = [e for e in all_expiries if 0 <= _dte(e) <= max_dte]

    if not valid_expiries:
        logger.info(
            "No expiries within %d DTE for %s (available: %s)",
            max_dte,
            symbol,
            list(all_expiries[:5]),
        )
        return None

    # Pick the nearest expiry.
    valid_expiries.sort(key=_dte)
    expiry = valid_expiries[0]
    dte = _dte(expiry)
    logger.debug("Selected expiry %s (%d DTE) for %s", expiry, dte, symbol)

    # Fetch the chain for that expiry.
    chain = ticker.option_chain(expiry)
    contracts: pd.DataFrame = (
        chain.calls if option_type == "call" else chain.puts
    )

    if contracts.empty:
        logger.info("Empty %s chain for %s exp %s", option_type, symbol, expiry)
        return None

    # Filter by open interest.
    contracts = contracts[
        contracts["openInterest"].fillna(0) >= min_open_interest
    ].copy()

    if contracts.empty:
        logger.info(
            "No %s contracts meet min open-interest=%d for %s exp %s",
            option_type,
            min_open_interest,
            symbol,
            expiry,
        )
        return None

    # Select strike nearest to spot (ATM).
    contracts["_dist"] = (contracts["strike"] - spot).abs()
    best = contracts.nsmallest(1, "_dist").iloc[0]

    contract = OptionContract(
        symbol=symbol,
        expiry=expiry,
        strike=float(best["strike"]),
        option_type=option_type,
        last_price=float(best.get("lastPrice", 0) or 0),
        bid=float(best.get("bid", 0) or 0),
        ask=float(best.get("ask", 0) or 0),
        implied_volatility=float(best.get("impliedVolatility", 0) or 0),
        open_interest=int(best.get("openInterest", 0) or 0),
        dte=dte,
    )
    logger.info(
        "Selected %s %s strike=%.2f exp=%s DTE=%d last=%.4f",
        symbol,
        option_type.upper(),
        contract.strike,
        expiry,
        dte,
        contract.last_price,
    )
    return contract
