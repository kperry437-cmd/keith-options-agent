"""
signals.py – SMA computation and golden/death cross detection.

A *golden cross* occurs when the faster SMA crosses *above* the slower SMA
compared to the previous bar.  A *death cross* is the opposite (crosses below).
Both require the previous bar's SMAs to have been in the opposite relationship,
ensuring we only fire on the actual crossover event rather than on every bar.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_GOLDEN = "golden"
_DEATH = "death"


def compute_smas(closes: pd.Series, periods: list[int]) -> dict[int, pd.Series]:
    """Return a mapping of {period: rolling-SMA Series} for *closes*."""
    return {p: closes.rolling(p, min_periods=p).mean() for p in periods}


def detect_cross(
    fast: pd.Series,
    slow: pd.Series,
) -> Optional[str]:
    """Detect a cross between *fast* and *slow* SMA series.

    Returns ``'golden'``, ``'death'``, or ``None`` (no cross on the last bar).

    A cross is only signalled when:
      - Both the current **and** prior values are non-NaN.
      - The relationship flips between the previous bar and the current bar.
    """
    if len(fast) < 2 or len(slow) < 2:
        return None

    prev_fast = fast.iloc[-2]
    prev_slow = slow.iloc[-2]
    curr_fast = fast.iloc[-1]
    curr_slow = slow.iloc[-1]

    if any(pd.isna(v) for v in (prev_fast, prev_slow, curr_fast, curr_slow)):
        return None

    was_below = prev_fast < prev_slow
    now_above = curr_fast > curr_slow

    was_above = prev_fast > prev_slow
    now_below = curr_fast < curr_slow

    if was_below and now_above:
        return _GOLDEN
    if was_above and now_below:
        return _DEATH
    return None


def get_signals(df: pd.DataFrame, cross_pairs: list[list[int]]) -> List[dict]:
    """Return a list of signal dicts for all crossing SMA pairs.

    Each signal dict contains:
        ``type``         – ``'golden'`` or ``'death'``
        ``fast_period``  – the faster SMA period
        ``slow_period``  – the slower SMA period

    Args:
        df:          DataFrame with a ``close`` column.
        cross_pairs: List of ``[fast_period, slow_period]`` pairs to monitor.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must have a 'close' column")

    all_periods: set[int] = set()
    for fast, slow in cross_pairs:
        all_periods.add(fast)
        all_periods.add(slow)

    smas = compute_smas(df["close"], list(all_periods))

    signals: List[dict] = []
    for fast_p, slow_p in cross_pairs:
        cross = detect_cross(smas[fast_p], smas[slow_p])
        if cross:
            logger.info(
                "Cross detected: %s (SMA%d / SMA%d)", cross, fast_p, slow_p
            )
            signals.append(
                {"type": cross, "fast_period": fast_p, "slow_period": slow_p}
            )

    return signals
