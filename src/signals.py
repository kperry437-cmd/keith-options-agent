"""
signals.py – Golden / death cross detection with a proximity filter.

A *golden cross* occurs when the fast SMA crosses **above** the slow SMA.
A *death cross*  occurs when the fast SMA crosses **below** the slow SMA.

The optional *proximity filter* suppresses a signal when the two SMAs are
already so far apart that the cross would be considered "stale" (e.g. the
fast SMA is more than ``proximity_pct`` percent away from the slow SMA at
the time of the cross).  This avoids chasing momentum that has already run.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Signal(str, Enum):
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    NONE = "none"


def detect_cross(
    df: pd.DataFrame,
    fast_col: str,
    slow_col: str,
    proximity_pct: float = 1.0,
) -> Signal:
    """Detect the most recent golden or death cross in *df*.

    Parameters
    ----------
    df:
        DataFrame containing at least *fast_col* and *slow_col* (SMA
        columns) as well as a ``close`` column.  Must be sorted in
        ascending time order.
    fast_col:
        Column name for the fast SMA (e.g. ``"sma_20"``).
    slow_col:
        Column name for the slow SMA (e.g. ``"sma_50"``).
    proximity_pct:
        Maximum allowed percentage distance between the two SMAs at the
        moment of the cross.  A value of ``1.0`` means 1 %.  Set to
        ``float("inf")`` to disable the filter entirely.

    Returns
    -------
    Signal
        ``Signal.GOLDEN_CROSS``, ``Signal.DEATH_CROSS``, or
        ``Signal.NONE`` if no cross was detected on the most recent bar.
    """
    required = {fast_col, slow_col, "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    clean = df[[fast_col, slow_col, "close"]].dropna()
    if len(clean) < 2:
        return Signal.NONE

    prev = clean.iloc[-2]
    curr = clean.iloc[-1]

    prev_fast, prev_slow = prev[fast_col], prev[slow_col]
    curr_fast, curr_slow = curr[fast_col], curr[slow_col]

    golden = prev_fast <= prev_slow and curr_fast > curr_slow
    death = prev_fast >= prev_slow and curr_fast < curr_slow

    if not (golden or death):
        return Signal.NONE

    # Proximity filter: reject if SMAs are too far apart at the cross point.
    mid = (curr_fast + curr_slow) / 2.0
    if mid == 0:
        return Signal.NONE
    spread_pct = abs(curr_fast - curr_slow) / mid * 100.0

    if spread_pct > proximity_pct:
        logger.debug(
            "Cross detected but suppressed by proximity filter "
            "(spread=%.3f%% > limit=%.3f%%)",
            spread_pct,
            proximity_pct,
        )
        return Signal.NONE

    result = Signal.GOLDEN_CROSS if golden else Signal.DEATH_CROSS
    logger.info(
        "%s detected (fast=%.4f slow=%.4f spread=%.3f%%)",
        result.value,
        curr_fast,
        curr_slow,
        spread_pct,
    )
    return result


def latest_signal(
    df: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    proximity_pct: float = 1.0,
) -> Signal:
    """Convenience wrapper that builds column names from period integers.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`src.data.get_candles_with_smas`.
    fast_period:
        Integer used to build ``"sma_{fast_period}"``.
    slow_period:
        Integer used to build ``"sma_{slow_period}"``.
    proximity_pct:
        Passed directly to :func:`detect_cross`.
    """
    return detect_cross(
        df,
        fast_col=f"sma_{fast_period}",
        slow_col=f"sma_{slow_period}",
        proximity_pct=proximity_pct,
    )
