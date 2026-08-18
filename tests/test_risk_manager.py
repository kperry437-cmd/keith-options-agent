"""Tests for RiskManager — strict risk rules."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.config import AgentConfig, OptionsConfig, RiskConfig
from src.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**risk_overrides) -> AgentConfig:
    risk_defaults = dict(
        max_position_pct=0.02,
        max_daily_loss_pct=0.05,
        max_open_positions=3,
        option_stop_loss_pct=0.50,
        min_dte=7,
        max_dte=45,
        max_delta=0.45,
        min_delta=0.25,
    )
    risk_defaults.update(risk_overrides)
    return AgentConfig(
        symbol="SPY",
        timeframe_minutes=5,
        sma_periods=[20, 50, 200],
        risk=RiskConfig(**risk_defaults),
        options=OptionsConfig(contract_quantity=1),
        market_open_hour=9,
        market_open_minute=30,
        market_close_hour=16,
        market_close_minute=0,
        pre_close_buffer_minutes=30,
        post_open_buffer_minutes=30,
    )


def _dt(hour: int, minute: int) -> datetime:
    """Return a timezone-aware datetime for today at hour:minute UTC."""
    return datetime.now(timezone.utc).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )


# ---------------------------------------------------------------------------
# is_trading_window
# ---------------------------------------------------------------------------

class TestTradingWindow:
    def test_midday_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.is_trading_window(_dt(12, 0)).allowed

    def test_before_market_open(self):
        rm = RiskManager(_make_config())
        assert not rm.is_trading_window(_dt(8, 0)).allowed

    def test_after_market_close(self):
        rm = RiskManager(_make_config())
        assert not rm.is_trading_window(_dt(17, 0)).allowed

    def test_within_post_open_buffer(self):
        rm = RiskManager(_make_config())
        # 9:45 is inside the 30-min post-open buffer (cutoff is 10:00)
        assert not rm.is_trading_window(_dt(9, 45)).allowed

    def test_exactly_at_open_buffer_edge(self):
        rm = RiskManager(_make_config())
        # 10:00 is exactly when the 30-min post-open buffer expires — allowed
        assert rm.is_trading_window(_dt(10, 0)).allowed

    def test_just_past_open_buffer(self):
        rm = RiskManager(_make_config())
        # 10:01 is one minute past the cutoff
        assert rm.is_trading_window(_dt(10, 1)).allowed

    def test_within_pre_close_buffer(self):
        rm = RiskManager(_make_config())
        # 15:35 is inside the 30-min pre-close buffer (cutoff is 15:30)
        assert not rm.is_trading_window(_dt(15, 35)).allowed

    def test_exactly_at_pre_close_cutoff(self):
        rm = RiskManager(_make_config())
        # 15:30 is exactly at the cutoff — blocked
        assert not rm.is_trading_window(_dt(15, 30)).allowed

    def test_one_minute_before_pre_close(self):
        rm = RiskManager(_make_config())
        # 15:29 is just before the pre-close buffer
        assert rm.is_trading_window(_dt(15, 29)).allowed

    def test_reason_string_present(self):
        rm = RiskManager(_make_config())
        result = rm.is_trading_window(_dt(8, 0))
        assert result.reason != ""


# ---------------------------------------------------------------------------
# check_daily_loss
# ---------------------------------------------------------------------------

class TestDailyLoss:
    def test_no_loss_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_daily_loss(0.0, 10_000).allowed

    def test_small_loss_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_daily_loss(-100, 10_000).allowed  # 1 %

    def test_at_threshold_blocked(self):
        rm = RiskManager(_make_config())
        # Exactly 5 % loss
        assert not rm.check_daily_loss(-500, 10_000).allowed

    def test_above_threshold_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_daily_loss(-600, 10_000).allowed

    def test_profit_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_daily_loss(500, 10_000).allowed

    def test_zero_portfolio_value_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_daily_loss(0, 0).allowed

    def test_negative_portfolio_value_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_daily_loss(0, -1000).allowed

    def test_custom_threshold(self):
        rm = RiskManager(_make_config(max_daily_loss_pct=0.10))
        assert rm.check_daily_loss(-900, 10_000).allowed      # 9 % — ok
        assert not rm.check_daily_loss(-1000, 10_000).allowed # 10 % — blocked


# ---------------------------------------------------------------------------
# check_open_positions
# ---------------------------------------------------------------------------

class TestOpenPositions:
    def test_zero_positions_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_open_positions(0).allowed

    def test_under_limit_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_open_positions(2).allowed

    def test_at_limit_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_open_positions(3).allowed

    def test_over_limit_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_open_positions(10).allowed

    def test_custom_limit(self):
        rm = RiskManager(_make_config(max_open_positions=1))
        assert rm.check_open_positions(0).allowed
        assert not rm.check_open_positions(1).allowed


# ---------------------------------------------------------------------------
# check_option_delta
# ---------------------------------------------------------------------------

class TestDelta:
    def test_midrange_delta_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_option_delta(0.35).allowed

    def test_below_min_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_option_delta(0.20).allowed

    def test_above_max_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_option_delta(0.50).allowed

    def test_at_min_boundary_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_option_delta(0.25).allowed

    def test_at_max_boundary_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_option_delta(0.45).allowed

    def test_negative_delta_uses_abs_value(self):
        rm = RiskManager(_make_config())
        # Put options have negative delta — magnitude should pass the same checks
        assert rm.check_option_delta(-0.35).allowed
        assert not rm.check_option_delta(-0.20).allowed
        assert not rm.check_option_delta(-0.50).allowed


# ---------------------------------------------------------------------------
# check_dte
# ---------------------------------------------------------------------------

class TestDTE:
    def test_midrange_dte_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_dte(21).allowed

    def test_below_min_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_dte(3).allowed

    def test_above_max_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_dte(60).allowed

    def test_at_min_boundary_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_dte(7).allowed

    def test_at_max_boundary_allowed(self):
        rm = RiskManager(_make_config())
        assert rm.check_dte(45).allowed

    def test_one_below_min_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_dte(6).allowed

    def test_one_above_max_blocked(self):
        rm = RiskManager(_make_config())
        assert not rm.check_dte(46).allowed


# ---------------------------------------------------------------------------
# calculate_position_size
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_standard_sizing(self):
        rm = RiskManager(_make_config())
        # max_spend = 100_000 * 0.02 = 2_000
        # cost_per_contract = 5.00 * 100 = 500
        # contracts = floor(2_000 / 500) = 4
        result = rm.calculate_position_size(100_000, 5.00)
        assert result.contracts == 4

    def test_returns_zero_when_budget_too_small(self):
        rm = RiskManager(_make_config())
        # max_spend = 5_000 * 0.02 = 100; cost_per_contract = 20 * 100 = 2_000 → 0 contracts
        result = rm.calculate_position_size(5_000, 20.00)
        assert result.contracts == 0
        assert result.max_loss == 0.0

    def test_zero_price_returns_zero(self):
        rm = RiskManager(_make_config())
        result = rm.calculate_position_size(100_000, 0)
        assert result.contracts == 0
        assert result.max_loss == 0.0

    def test_max_loss_is_fifty_percent_of_cost(self):
        rm = RiskManager(_make_config())
        result = rm.calculate_position_size(100_000, 5.00)
        expected = result.contracts * 5.00 * 100 * 0.50
        assert abs(result.max_loss - expected) < 0.01

    def test_larger_portfolio_more_contracts(self):
        rm = RiskManager(_make_config())
        small = rm.calculate_position_size(10_000, 2.00)
        large = rm.calculate_position_size(100_000, 2.00)
        assert large.contracts > small.contracts


# ---------------------------------------------------------------------------
# should_stop_loss
# ---------------------------------------------------------------------------

class TestStopLoss:
    def test_below_threshold_no_stop(self):
        rm = RiskManager(_make_config())
        # 40 % loss — threshold is 50 %
        assert not rm.should_stop_loss(5.00, 3.00)

    def test_exactly_at_threshold_triggers_stop(self):
        rm = RiskManager(_make_config())
        assert rm.should_stop_loss(5.00, 2.50)

    def test_above_threshold_triggers_stop(self):
        rm = RiskManager(_make_config())
        assert rm.should_stop_loss(5.00, 2.00)

    def test_no_change_no_stop(self):
        rm = RiskManager(_make_config())
        assert not rm.should_stop_loss(5.00, 5.00)

    def test_gain_no_stop(self):
        rm = RiskManager(_make_config())
        assert not rm.should_stop_loss(5.00, 8.00)

    def test_zero_entry_price_no_stop(self):
        rm = RiskManager(_make_config())
        assert not rm.should_stop_loss(0.0, 5.00)

    def test_custom_stop_threshold(self):
        rm = RiskManager(_make_config(option_stop_loss_pct=0.25))
        assert rm.should_stop_loss(4.00, 2.99)    # > 25 % loss
        assert not rm.should_stop_loss(4.00, 3.01) # < 25 % loss


# ---------------------------------------------------------------------------
# approve_trade (integration)
# ---------------------------------------------------------------------------

class TestApproveTrade:
    def _good_kwargs(self) -> dict:
        return dict(
            now=_dt(12, 0),
            daily_pnl=100.0,
            portfolio_value=100_000,
            open_position_count=1,
            option_delta=0.35,
            dte=21,
        )

    def test_all_checks_pass(self):
        rm = RiskManager(_make_config())
        assert rm.approve_trade(**self._good_kwargs()).allowed

    def test_rejects_outside_window(self):
        rm = RiskManager(_make_config())
        kwargs = {**self._good_kwargs(), "now": _dt(8, 0)}
        assert not rm.approve_trade(**kwargs).allowed

    def test_rejects_excessive_daily_loss(self):
        rm = RiskManager(_make_config())
        kwargs = {**self._good_kwargs(), "daily_pnl": -6_000}
        assert not rm.approve_trade(**kwargs).allowed

    def test_rejects_max_positions(self):
        rm = RiskManager(_make_config())
        kwargs = {**self._good_kwargs(), "open_position_count": 3}
        assert not rm.approve_trade(**kwargs).allowed

    def test_rejects_bad_delta(self):
        rm = RiskManager(_make_config())
        kwargs = {**self._good_kwargs(), "option_delta": 0.10}
        assert not rm.approve_trade(**kwargs).allowed

    def test_rejects_bad_dte(self):
        rm = RiskManager(_make_config())
        kwargs = {**self._good_kwargs(), "dte": 3}
        assert not rm.approve_trade(**kwargs).allowed

    def test_result_has_reason_on_rejection(self):
        rm = RiskManager(_make_config())
        kwargs = {**self._good_kwargs(), "dte": 3}
        result = rm.approve_trade(**kwargs)
        assert not result.allowed
        assert result.reason != ""
