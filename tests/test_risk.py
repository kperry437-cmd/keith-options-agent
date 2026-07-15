"""Unit tests for the risk management module."""

import pytest
from datetime import date

from src.risk import RiskManager, OpenPosition


def _make_risk_manager(**overrides):
    defaults = dict(
        max_portfolio_risk_pct=0.02,
        max_position_risk_pct=0.01,
        max_open_positions=5,
        stop_loss_pct=0.50,
        take_profit_pct=1.00,
        max_daily_loss_pct=0.05,
    )
    defaults.update(overrides)
    return RiskManager(**defaults)


def _make_position(symbol="SPY240119C00480000", underlying="SPY", option_type="call",
                   contracts=1, entry_price=5.00):
    sl = entry_price * 0.50
    tp = entry_price * 2.00
    return OpenPosition(
        symbol=symbol,
        underlying=underlying,
        option_type=option_type,
        contracts=contracts,
        entry_price=entry_price,
        stop_loss_price=sl,
        take_profit_price=tp,
    )


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_basic_sizing(self):
        rm = _make_risk_manager(max_position_risk_pct=0.01)
        # 1% of 100_000 = 1_000; contracts = floor(1000 / (5*100)) = 2
        n = rm.size_position(portfolio_value=100_000, option_premium=5.00)
        assert n == 2

    def test_zero_premium_returns_zero(self):
        rm = _make_risk_manager()
        assert rm.size_position(100_000, 0.0) == 0

    def test_negative_premium_returns_zero(self):
        rm = _make_risk_manager()
        assert rm.size_position(100_000, -1.0) == 0

    def test_zero_portfolio_returns_zero(self):
        rm = _make_risk_manager()
        assert rm.size_position(0, 5.0) == 0

    def test_high_premium_may_return_zero(self):
        rm = _make_risk_manager(max_position_risk_pct=0.001)
        # 0.1% of 10_000 = 10; contracts = floor(10 / (50*100)) = 0
        n = rm.size_position(10_000, 50.00)
        assert n == 0


# ---------------------------------------------------------------------------
# Stop/take-profit computation
# ---------------------------------------------------------------------------

class TestComputeStops:
    def test_stop_loss_below_entry(self):
        rm = _make_risk_manager(stop_loss_pct=0.50)
        sl, tp = rm.compute_stops(10.0, "call")
        assert sl == pytest.approx(5.0)

    def test_take_profit_above_entry(self):
        rm = _make_risk_manager(take_profit_pct=1.00)
        sl, tp = rm.compute_stops(10.0, "call")
        assert tp == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# can_open_position
# ---------------------------------------------------------------------------

class TestCanOpenPosition:
    def test_can_open_when_empty(self):
        rm = _make_risk_manager()
        assert rm.can_open_position("SPY") is True

    def test_cannot_open_duplicate_underlying(self):
        rm = _make_risk_manager()
        pos = _make_position()
        rm.open_position(pos)
        assert rm.can_open_position("SPY") is False

    def test_cannot_open_when_max_reached(self):
        rm = _make_risk_manager(max_open_positions=2)
        rm.open_position(_make_position("SYM1", "SPY"))
        rm.open_position(_make_position("SYM2", "QQQ"))
        assert rm.can_open_position("AAPL") is False

    def test_can_open_different_underlying(self):
        rm = _make_risk_manager(max_open_positions=5)
        rm.open_position(_make_position("SYM1", "SPY"))
        assert rm.can_open_position("QQQ") is True


# ---------------------------------------------------------------------------
# open / close positions
# ---------------------------------------------------------------------------

class TestPositionLifecycle:
    def test_open_adds_to_positions(self):
        rm = _make_risk_manager()
        pos = _make_position()
        rm.open_position(pos)
        assert "SPY240119C00480000" in rm.open_positions

    def test_close_removes_position(self):
        rm = _make_risk_manager()
        pos = _make_position()
        rm.open_position(pos)
        rm.close_position("SPY240119C00480000", exit_price=7.50)
        assert "SPY240119C00480000" not in rm.open_positions

    def test_close_returns_pnl(self):
        rm = _make_risk_manager()
        pos = _make_position(entry_price=5.00, contracts=2)
        rm.open_position(pos)
        pnl = rm.close_position("SPY240119C00480000", exit_price=7.50)
        # (7.50 - 5.00) * 100 * 2 = 500
        assert pnl == pytest.approx(500.0)

    def test_close_unknown_symbol_returns_none(self):
        rm = _make_risk_manager()
        result = rm.close_position("NONEXISTENT", 5.0)
        assert result is None

    def test_negative_pnl_on_loss(self):
        rm = _make_risk_manager()
        pos = _make_position(entry_price=5.00, contracts=1)
        rm.open_position(pos)
        pnl = rm.close_position("SPY240119C00480000", exit_price=2.50)
        assert pnl == pytest.approx(-250.0)


# ---------------------------------------------------------------------------
# Cost basis
# ---------------------------------------------------------------------------

class TestCostBasis:
    def test_cost_basis(self):
        pos = _make_position(entry_price=3.00, contracts=4)
        # 3.00 * 100 * 4 = 1200
        assert pos.cost_basis == pytest.approx(1200.0)


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------

class TestDailyLossLimit:
    def test_not_reached_initially(self):
        rm = _make_risk_manager(max_daily_loss_pct=0.05)
        assert rm.daily_loss_limit_reached(100_000) is False

    def test_reached_after_large_loss(self):
        rm = _make_risk_manager(max_daily_loss_pct=0.05)
        rm.record_trade_pnl(-6_000)  # 6% loss on 100k portfolio
        assert rm.daily_loss_limit_reached(100_000) is True

    def test_not_reached_below_threshold(self):
        rm = _make_risk_manager(max_daily_loss_pct=0.05)
        rm.record_trade_pnl(-4_000)  # 4% — below 5% cap
        assert rm.daily_loss_limit_reached(100_000) is False

    def test_zero_portfolio_value_returns_false(self):
        rm = _make_risk_manager()
        rm.record_trade_pnl(-1_000)
        assert rm.daily_loss_limit_reached(0) is False


# ---------------------------------------------------------------------------
# positions_to_close
# ---------------------------------------------------------------------------

class TestPositionsToClose:
    def test_stop_loss_triggered(self):
        rm = _make_risk_manager()
        pos = _make_position(entry_price=10.00, contracts=1)
        # stop_loss_price = 5.00
        rm.open_position(pos)
        result = rm.positions_to_close({"SPY240119C00480000": 4.99})
        assert "SPY240119C00480000" in result
        assert result["SPY240119C00480000"] == "stop_loss"

    def test_take_profit_triggered(self):
        rm = _make_risk_manager()
        pos = _make_position(entry_price=10.00, contracts=1)
        # take_profit_price = 20.00
        rm.open_position(pos)
        result = rm.positions_to_close({"SPY240119C00480000": 20.00})
        assert "SPY240119C00480000" in result
        assert result["SPY240119C00480000"] == "take_profit"

    def test_neither_triggered(self):
        rm = _make_risk_manager()
        pos = _make_position(entry_price=10.00, contracts=1)
        rm.open_position(pos)
        result = rm.positions_to_close({"SPY240119C00480000": 12.00})
        assert result == {}

    def test_missing_price_skipped(self):
        rm = _make_risk_manager()
        pos = _make_position(entry_price=10.00, contracts=1)
        rm.open_position(pos)
        result = rm.positions_to_close({})
        assert result == {}
