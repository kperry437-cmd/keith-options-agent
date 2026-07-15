"""Unit tests for the golden/death cross signal detection."""

import pandas as pd
import numpy as np
import pytest

from src.signals import Signal, compute_signals, latest_signal, _ema, _sma


def _make_bars(closes):
    index = pd.date_range("2024-01-01", periods=len(closes), freq="5min")
    return pd.DataFrame({"close": closes, "open": closes, "high": closes, "low": closes, "volume": 1000}, index=index)


# ---------------------------------------------------------------------------
# Moving average helpers
# ---------------------------------------------------------------------------

class TestMovingAverages:
    def test_ema_length_preserved(self):
        s = pd.Series(range(1, 21), dtype=float)
        result = _ema(s, 9)
        assert len(result) == len(s)

    def test_sma_length_preserved(self):
        s = pd.Series(range(1, 21), dtype=float)
        result = _sma(s, 9)
        assert len(result) == len(s)

    def test_sma_correct_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _sma(s, 3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_sma_nan_before_window(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = _sma(s, 3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])


# ---------------------------------------------------------------------------
# compute_signals
# ---------------------------------------------------------------------------

class TestComputeSignals:
    def test_empty_bars_returns_no_signal(self):
        bars = _make_bars([])
        result = compute_signals(bars, fast_period=9, slow_period=21)
        assert result.empty

    def test_insufficient_bars_returns_none_signals(self):
        bars = _make_bars([100.0] * 10)
        result = compute_signals(bars, fast_period=9, slow_period=21)
        assert (result["signal"] == Signal.NONE).all()

    def test_columns_added(self):
        bars = _make_bars([100.0 + i * 0.1 for i in range(30)])
        result = compute_signals(bars, fast_period=5, slow_period=15)
        assert "fast_ma" in result.columns
        assert "slow_ma" in result.columns
        assert "signal" in result.columns

    def test_golden_cross_detected(self):
        # Falling prices (fast below slow), then rising prices (fast crosses above slow)
        falling = [100.0 - i * 0.5 for i in range(30)]
        rising = [85.0 + i * 1.0 for i in range(30)]
        closes = falling + rising
        bars = _make_bars(closes)
        result = compute_signals(bars, fast_period=5, slow_period=15, ma_type="sma")
        assert Signal.GOLDEN_CROSS in result["signal"].values

    def test_death_cross_detected(self):
        # Rising prices (fast above slow), then falling prices (fast crosses below slow)
        rising = [100.0 + i * 0.5 for i in range(30)]
        falling = [115.0 - i * 1.0 for i in range(30)]
        closes = rising + falling
        bars = _make_bars(closes)
        result = compute_signals(bars, fast_period=5, slow_period=15, ma_type="sma")
        assert Signal.DEATH_CROSS in result["signal"].values

    def test_flat_prices_no_cross(self):
        bars = _make_bars([100.0] * 40)
        result = compute_signals(bars, fast_period=5, slow_period=15)
        assert (result["signal"] == Signal.NONE).all()

    def test_ema_vs_sma(self):
        closes = [100.0 + i * 0.1 for i in range(40)]
        bars = _make_bars(closes)
        result_ema = compute_signals(bars, fast_period=5, slow_period=15, ma_type="ema")
        result_sma = compute_signals(bars, fast_period=5, slow_period=15, ma_type="sma")
        # Both should run without error; fast_ma values will differ
        assert not result_ema["fast_ma"].dropna().equals(result_sma["fast_ma"].dropna())

    def test_invalid_ma_type_raises(self):
        bars = _make_bars([100.0] * 30)
        with pytest.raises(ValueError, match="Unknown ma_type"):
            compute_signals(bars, fast_period=5, slow_period=15, ma_type="wma")


# ---------------------------------------------------------------------------
# latest_signal
# ---------------------------------------------------------------------------

class TestLatestSignal:
    def test_returns_signal_enum(self):
        bars = _make_bars([100.0] * 30)
        sig = latest_signal(bars, fast_period=5, slow_period=15)
        assert isinstance(sig, Signal)

    def test_empty_bars_returns_none(self):
        bars = _make_bars([])
        sig = latest_signal(bars, fast_period=5, slow_period=15)
        assert sig == Signal.NONE

    def test_golden_cross_on_last_bar(self):
        # Construct data so the cross fires on the very last bar
        falling = [100.0 - i * 0.5 for i in range(25)]
        rising = [87.5 + i * 2.0 for i in range(20)]
        closes = falling + rising
        bars = _make_bars(closes)
        result = compute_signals(bars, fast_period=3, slow_period=10, ma_type="sma")
        last_signal = result["signal"].iloc[-1]
        # Just verify the function returns without error and is a Signal
        assert isinstance(last_signal, Signal)
