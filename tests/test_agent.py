"""Unit tests for the options agent modules."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# src/signals.py
# ---------------------------------------------------------------------------
from src.signals import Signal, detect_cross, latest_signal


def _make_df(fast_vals, slow_vals, close_vals=None):
    n = len(fast_vals)
    if close_vals is None:
        close_vals = [100.0] * n
    return pd.DataFrame(
        {
            "close": close_vals,
            "sma_20": fast_vals,
            "sma_50": slow_vals,
        }
    )


class TestDetectCross:
    def test_golden_cross(self):
        df = _make_df([99.0, 101.0], [100.0, 100.0])
        assert detect_cross(df, "sma_20", "sma_50", proximity_pct=100.0) == Signal.GOLDEN_CROSS

    def test_death_cross(self):
        df = _make_df([101.0, 99.0], [100.0, 100.0])
        assert detect_cross(df, "sma_20", "sma_50", proximity_pct=100.0) == Signal.DEATH_CROSS

    def test_no_cross(self):
        df = _make_df([99.0, 99.5], [100.0, 100.0])
        assert detect_cross(df, "sma_20", "sma_50", proximity_pct=100.0) == Signal.NONE

    def test_proximity_filter_suppresses(self):
        # Spread = 10 % between 90 and 110 → filtered when limit = 1 %
        df = _make_df([89.0, 91.0], [100.0, 100.0])
        assert detect_cross(df, "sma_20", "sma_50", proximity_pct=1.0) == Signal.NONE

    def test_proximity_filter_passes(self):
        # Very small spread, within 1 %
        df = _make_df([99.9, 100.1], [100.0, 100.0])
        result = detect_cross(df, "sma_20", "sma_50", proximity_pct=1.0)
        assert result == Signal.GOLDEN_CROSS

    def test_not_enough_rows(self):
        df = _make_df([100.0], [100.0])
        assert detect_cross(df, "sma_20", "sma_50") == Signal.NONE

    def test_missing_column_raises(self):
        df = pd.DataFrame({"close": [1.0, 2.0], "sma_20": [1.0, 2.0]})
        with pytest.raises(ValueError, match="missing columns"):
            detect_cross(df, "sma_20", "sma_50")

    def test_latest_signal_wrapper(self):
        df = _make_df([99.0, 101.0], [100.0, 100.0])
        assert latest_signal(df, proximity_pct=100.0) == Signal.GOLDEN_CROSS


# ---------------------------------------------------------------------------
# src/data.py  (SMA logic only – no network calls)
# ---------------------------------------------------------------------------
from src.data import add_sma


class TestAddSma:
    def test_columns_added(self):
        df = pd.DataFrame({"close": list(range(1, 101))})
        result = add_sma(df, fast_period=10, slow_period=20)
        assert "sma_10" in result.columns
        assert "sma_20" in result.columns

    def test_sma_values(self):
        df = pd.DataFrame({"close": [1.0] * 50})
        result = add_sma(df, fast_period=5, slow_period=10)
        assert result["sma_5"].dropna().iloc[-1] == pytest.approx(1.0)
        assert result["sma_10"].dropna().iloc[-1] == pytest.approx(1.0)

    def test_original_not_mutated(self):
        df = pd.DataFrame({"close": [1.0] * 30})
        add_sma(df, 10, 20)
        assert "sma_10" not in df.columns


# ---------------------------------------------------------------------------
# src/risk.py
# ---------------------------------------------------------------------------
from src.risk import PositionRisk, RiskParams, compute_risk, should_exit


class TestComputeRisk:
    def _params(self):
        return RiskParams(size_pct=0.10, stop_pct=0.05, target_pct=0.25, max_hold_days=1)

    def test_stop_price(self):
        risk = compute_risk(2.0, 10_000, self._params())
        assert risk.stop_price == pytest.approx(2.0 * 0.95)

    def test_target_price(self):
        risk = compute_risk(2.0, 10_000, self._params())
        assert risk.target_price == pytest.approx(2.0 * 1.25)

    def test_contracts(self):
        # budget = 0.10 * 10_000 = 1000; contract cost = 2.0 * 100 = 200 → 5
        risk = compute_risk(2.0, 10_000, self._params())
        assert risk.contracts == 5

    def test_notional(self):
        risk = compute_risk(2.0, 10_000, self._params())
        assert risk.notional == pytest.approx(2.0 * 5 * 100)

    def test_max_exit_time(self):
        entry = datetime(2024, 1, 10, tzinfo=timezone.utc)
        risk = compute_risk(1.0, 10_000, self._params(), entry_time=entry)
        assert risk.max_exit_time == entry + timedelta(days=1)

    def test_invalid_premium_raises(self):
        with pytest.raises(ValueError):
            compute_risk(0.0, 10_000)

    def test_invalid_equity_raises(self):
        with pytest.raises(ValueError):
            compute_risk(1.0, 0.0)

    def test_at_least_one_contract(self):
        # tiny equity → should still get 1 contract
        risk = compute_risk(100.0, 1.0, self._params())
        assert risk.contracts >= 1


class TestShouldExit:
    def _risk(self, entry=2.0):
        params = RiskParams(stop_pct=0.05, target_pct=0.25, max_hold_days=1)
        return compute_risk(entry, 10_000, params)

    def test_stop_triggered(self):
        risk = self._risk()
        flag, reason = should_exit(risk.stop_price - 0.01, risk)
        assert flag is True
        assert "stop-loss" in reason

    def test_target_triggered(self):
        risk = self._risk()
        flag, reason = should_exit(risk.target_price + 0.01, risk)
        assert flag is True
        assert "take-profit" in reason

    def test_max_hold_triggered(self):
        risk = self._risk()
        future = risk.max_exit_time + timedelta(seconds=1)
        flag, reason = should_exit(risk.entry_premium, risk, now=future)
        assert flag is True
        assert "max-hold" in reason

    def test_hold(self):
        risk = self._risk()
        mid = (risk.stop_price + risk.target_price) / 2.0
        past = risk.max_exit_time - timedelta(hours=1)
        flag, reason = should_exit(mid, risk, now=past)
        assert flag is False
        assert reason == "hold"


# ---------------------------------------------------------------------------
# src/state.py
# ---------------------------------------------------------------------------
from src.state import (
    AgentState,
    PositionState,
    clear_position,
    load_state,
    save_state,
)


class TestState:
    def _sample_state(self):
        pos = PositionState(
            symbol="SPY",
            expiry="2024-01-15",
            strike=450.0,
            option_type="call",
            entry_premium=2.50,
            contracts=3,
            notional=750.0,
            stop_price=2.375,
            target_price=3.125,
            max_exit_time="2024-01-11T00:00:00+00:00",
            entry_time="2024-01-10T15:00:00+00:00",
        )
        return AgentState(
            last_run="2024-01-10T15:00:00+00:00",
            run_count=5,
            last_signal="golden_cross",
            position=pos,
        )

    def test_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state = self._sample_state()
            save_state(state, path)
            loaded = load_state(path)
            assert loaded.run_count == 5
            assert loaded.last_signal == "golden_cross"
            assert loaded.position is not None
            assert loaded.position.symbol == "SPY"
            assert loaded.position.strike == 450.0
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        state = load_state("/tmp/nonexistent_state_abc123.json")
        assert state.run_count == 0
        assert state.position is None

    def test_corrupt_file_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            f.write("not valid json {{")
            path = f.name
        try:
            state = load_state(path)
            assert state.run_count == 0
        finally:
            os.unlink(path)

    def test_clear_position(self):
        state = self._sample_state()
        cleared = clear_position(state)
        assert cleared.position is None
        assert cleared.run_count == state.run_count

    def test_no_position_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state = AgentState(run_count=3, last_signal="none")
            save_state(state, path)
            loaded = load_state(path)
            assert loaded.position is None
            assert loaded.run_count == 3
        finally:
            os.unlink(path)
