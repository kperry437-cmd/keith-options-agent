"""Golden-cross / death-cross signal detection on 5-minute bars."""

import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Signal(Enum):
    GOLDEN_CROSS = "golden_cross"   # fast MA crosses above slow MA → bullish
    DEATH_CROSS = "death_cross"     # fast MA crosses below slow MA → bearish
    NONE = "none"


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _compute_ma(series: pd.Series, period: int, ma_type: str) -> pd.Series:
    if ma_type == "ema":
        return _ema(series, period)
    if ma_type == "sma":
        return _sma(series, period)
    raise ValueError(f"Unknown ma_type '{ma_type}'. Use 'ema' or 'sma'.")


def compute_signals(
    bars: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    ma_type: str = "ema",
    confirmation_bars: int = 1,
) -> pd.DataFrame:
    """Compute fast/slow moving averages and cross signals for every bar.

    Parameters
    ----------
    bars : DataFrame with a 'close' column, indexed by datetime.
    fast_period : Number of bars for the fast moving average.
    slow_period : Number of bars for the slow moving average.
    ma_type : 'ema' or 'sma'.
    confirmation_bars : How many consecutive bars the cross must hold to fire.

    Returns
    -------
    DataFrame with added columns:
        fast_ma, slow_ma, signal (Signal enum value)
    """
    if bars.empty or len(bars) < slow_period:
        bars = bars.copy()
        bars["fast_ma"] = np.nan
        bars["slow_ma"] = np.nan
        bars["signal"] = Signal.NONE
        return bars

    df = bars.copy()
    df["fast_ma"] = _compute_ma(df["close"], fast_period, ma_type)
    df["slow_ma"] = _compute_ma(df["close"], slow_period, ma_type)

    # Difference: positive when fast > slow
    df["_diff"] = df["fast_ma"] - df["slow_ma"]
    df["_prev_diff"] = df["_diff"].shift(1)

    df["signal"] = Signal.NONE

    if confirmation_bars <= 1:
        # Simple single-bar cross detection
        golden_mask = (df["_prev_diff"] < 0) & (df["_diff"] > 0)
        death_mask = (df["_prev_diff"] > 0) & (df["_diff"] < 0)
    else:
        # Require `confirmation_bars` consecutive bars above/below zero after the cross
        golden_mask = pd.Series(False, index=df.index)
        death_mask = pd.Series(False, index=df.index)

        for i in range(confirmation_bars, len(df)):
            cross_idx = i - confirmation_bars
            if df["_prev_diff"].iloc[cross_idx] < 0 and df["_diff"].iloc[cross_idx] > 0:
                # Check all confirmation bars are still positive
                if (df["_diff"].iloc[cross_idx : i + 1] > 0).all():
                    golden_mask.iloc[i] = True
            if df["_prev_diff"].iloc[cross_idx] > 0 and df["_diff"].iloc[cross_idx] < 0:
                if (df["_diff"].iloc[cross_idx : i + 1] < 0).all():
                    death_mask.iloc[i] = True

    df.loc[golden_mask, "signal"] = Signal.GOLDEN_CROSS
    df.loc[death_mask, "signal"] = Signal.DEATH_CROSS

    df = df.drop(columns=["_diff", "_prev_diff"])
    return df


def latest_signal(
    bars: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    ma_type: str = "ema",
    confirmation_bars: int = 1,
) -> Signal:
    """Return the signal on the most recent completed bar."""
    df = compute_signals(bars, fast_period, slow_period, ma_type, confirmation_bars)
    if df.empty:
        return Signal.NONE
    return df["signal"].iloc[-1]
