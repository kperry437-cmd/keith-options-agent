"""
test_signals.py – Unit tests for SMA computation and cross detection.
"""

import numpy as np
import pandas as pd
import pytest

from signals import compute_smas, detect_cross, get_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(closes: list[float]) -> pd.DataFrame:
    """Build a minimal DataFrame with a 'close' column."""
    return pd.DataFrame({"close": closes})


def _make_series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# compute_smas
# ---------------------------------------------------------------------------


class TestComputeSmas:
    def test_returns_all_periods(self):
        df = _make_df(list(range(1, 51)))
        smas = compute_smas(df["close"], [10, 20])
        assert set(smas.keys()) == {10, 20}

    def test_sma_values_correct(self):
        closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        smas = compute_smas(closes, [3])
        # SMA(3) over last 3 values: (3+4+5)/3 = 4.0
        assert smas[3].iloc[-1] == pytest.approx(4.0)

    def test_insufficient_data_yields_nan(self):
        closes = pd.Series([1.0, 2.0])
        smas = compute_smas(closes, [5])
        assert pd.isna(smas[5].iloc[-1])


# ---------------------------------------------------------------------------
# detect_cross
# ---------------------------------------------------------------------------


class TestDetectCross:
    def test_golden_cross(self):
        # fast was below slow, now above
        fast = _make_series([9.0, 11.0])
        slow = _make_series([10.0, 10.0])
        assert detect_cross(fast, slow) == "golden"

    def test_death_cross(self):
        # fast was above slow, now below
        fast = _make_series([11.0, 9.0])
        slow = _make_series([10.0, 10.0])
        assert detect_cross(fast, slow) == "death"

    def test_no_cross_above(self):
        # fast stays above slow throughout
        fast = _make_series([11.0, 12.0])
        slow = _make_series([10.0, 10.0])
        assert detect_cross(fast, slow) is None

    def test_no_cross_below(self):
        # fast stays below slow throughout
        fast = _make_series([9.0, 8.0])
        slow = _make_series([10.0, 10.0])
        assert detect_cross(fast, slow) is None

    def test_exactly_equal_no_cross(self):
        # fast equals slow on both bars → no cross
        fast = _make_series([10.0, 10.0])
        slow = _make_series([10.0, 10.0])
        assert detect_cross(fast, slow) is None

    def test_nan_prev_returns_none(self):
        fast = _make_series([float("nan"), 11.0])
        slow = _make_series([10.0, 10.0])
        assert detect_cross(fast, slow) is None

    def test_nan_curr_returns_none(self):
        fast = _make_series([9.0, float("nan")])
        slow = _make_series([10.0, 10.0])
        assert detect_cross(fast, slow) is None

    def test_too_few_bars_returns_none(self):
        assert detect_cross(_make_series([10.0]), _make_series([10.0])) is None
        assert detect_cross(_make_series([]), _make_series([])) is None


# ---------------------------------------------------------------------------
# get_signals
# ---------------------------------------------------------------------------


class TestGetSignals:
    def test_no_signal_on_flat_prices(self):
        df = _make_df([100.0] * 210)
        signals = get_signals(df, [[20, 50], [50, 200]])
        assert signals == []

    def test_missing_close_column_raises(self):
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="close"):
            get_signals(df, [[20, 50]])

    def test_signal_has_required_keys(self):
        # Craft a minimal golden-cross scenario for SMA(2) / SMA(3)
        closes = [10.0, 10.0, 10.0, 5.0, 5.0, 20.0]  # last bar spikes above
        df = _make_df(closes)
        signals = get_signals(df, [[2, 3]])
        # May or may not cross depending on values; just validate structure when present
        for s in signals:
            assert "type" in s
            assert "fast_period" in s
            assert "slow_period" in s
            assert s["type"] in ("golden", "death")

    def test_golden_cross_detected(self):
        # SMA(2) vs SMA(3): engineer a strict cross on the last bar.
        # Bar -2: SMA(2)=mean(6,6)=6.0, SMA(3)=mean(8,6,6)=6.67 → fast < slow ✓
        # Bar -1: SMA(2)=mean(6,20)=13.0, SMA(3)=mean(6,6,20)=10.67 → fast > slow ✓
        closes = [10.0, 8.0, 6.0, 6.0, 20.0]
        df = _make_df(closes)
        signals = get_signals(df, [[2, 3]])
        golden = [s for s in signals if s["type"] == "golden"]
        assert len(golden) == 1

    def test_death_cross_detected(self):
        # Bar -2: SMA(2)=mean(9,9)=9.0, SMA(3)=mean(7,9,9)=8.33 → fast > slow ✓
        # Bar -1: SMA(2)=mean(9,2)=5.5, SMA(3)=mean(9,9,2)=6.67 → fast < slow ✓
        closes = [5.0, 7.0, 9.0, 9.0, 2.0]
        df = _make_df(closes)
        signals = get_signals(df, [[2, 3]])
        death = [s for s in signals if s["type"] == "death"]
        assert len(death) == 1

    def test_empty_cross_pairs(self):
        df = _make_df([100.0] * 210)
        assert get_signals(df, []) == []
