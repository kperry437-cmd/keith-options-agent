"""
data.py – Fetch 5-minute OHLCV candles and compute SMAs.

Uses yfinance for price data.  The SMA periods are configurable via
``config.json``; defaults are 20 and 50 periods.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_candles(
    symbol: str,
    period: str = "5d",
    interval: str = "5m",
) -> pd.DataFrame:
    """Download OHLCV candles for *symbol*.

    Parameters
    ----------
    symbol:
        Ticker symbol, e.g. ``"SPY"``.
    period:
        Look-back window understood by yfinance (e.g. ``"5d"``).
    interval:
        Bar width understood by yfinance (e.g. ``"5m"``).

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.
        Index: timezone-aware datetime (UTC).
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No candle data returned for {symbol!r}")

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.sort_index(inplace=True)
    logger.debug("Fetched %d candles for %s", len(df), symbol)
    return df


def add_sma(
    df: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
) -> pd.DataFrame:
    """Return a copy of *df* with ``sma_fast`` and ``sma_slow`` columns appended.

    Parameters
    ----------
    df:
        DataFrame with at least a ``close`` column.
    fast_period:
        Rolling window for the fast SMA.
    slow_period:
        Rolling window for the slow SMA.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with two new columns.
    """
    out = df.copy()
    out[f"sma_{fast_period}"] = out["close"].rolling(fast_period).mean()
    out[f"sma_{slow_period}"] = out["close"].rolling(slow_period).mean()
    return out


def get_candles_with_smas(
    symbol: str,
    fast_period: int = 20,
    slow_period: int = 50,
    period: str = "5d",
    interval: str = "5m",
) -> pd.DataFrame:
    """Convenience wrapper: fetch candles **and** attach SMAs.

    Returns
    -------
    pd.DataFrame
        Candle data enriched with ``sma_{fast_period}`` and
        ``sma_{slow_period}`` columns.
    """
    df = fetch_candles(symbol, period=period, interval=interval)
    df = add_sma(df, fast_period=fast_period, slow_period=slow_period)
    return df
