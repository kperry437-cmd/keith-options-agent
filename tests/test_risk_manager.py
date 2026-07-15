"""Unit tests for RiskManager."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import AgentConfig
from src.risk_manager import RiskManager


@pytest.fixture
def cfg() -> AgentConfig:
    return AgentConfig(
        max_allocation_pct=0.15,
        max_loss_pct=0.05,
        take_profit_pct=0.25,
        max_hold_minutes=390,
    )


@pytest.fixture
def rm(cfg: AgentConfig) -> RiskManager:
    return RiskManager(cfg)


# ---------------------------------------------------------------------------
# AgentConfig validation
# ---------------------------------------------------------------------------

class TestAgentConfigValidation:
    def test_valid_config(self):
        cfg = AgentConfig(0.15, 0.05, 0.25, 390)
        assert cfg.max_allocation_pct == 0.15

    def test_invalid_allocation(self):
        with pytest.raises(ValueError):
            AgentConfig(0.0, 0.05, 0.25, 390)

    def test_invalid_max_loss(self):
        with pytest.raises(ValueError):
            AgentConfig(0.15, 0.0, 0.25, 390)

    def test_invalid_take_profit(self):
        with pytest.raises(ValueError):
            AgentConfig(0.15, 0.05, 0.0, 390)

    def test_invalid_hold_minutes(self):
        with pytest.raises(ValueError):
            AgentConfig(0.15, 0.05, 0.25, 0)


# ---------------------------------------------------------------------------
# max_position_value
# ---------------------------------------------------------------------------

class TestMaxPositionValue:
    def test_basic(self, rm: RiskManager):
        # 15% of $100,000 = $15,000
        assert rm.max_position_value(100_000) == pytest.approx(15_000)

    def test_small_portfolio(self, rm: RiskManager):
        assert rm.max_position_value(1_000) == pytest.approx(150)

    def test_zero_equity_raises(self, rm: RiskManager):
        with pytest.raises(ValueError):
            rm.max_position_value(0)

    def test_negative_equity_raises(self, rm: RiskManager):
        with pytest.raises(ValueError):
            rm.max_position_value(-5_000)


# ---------------------------------------------------------------------------
# position_contracts
# ---------------------------------------------------------------------------

class TestPositionContracts:
    def test_standard(self, rm: RiskManager):
        # $15,000 cap / ($2.50 * 100) = 60 contracts
        assert rm.position_contracts(100_000, 2.50) == 60

    def test_fractional_rounds_down(self, rm: RiskManager):
        # $15,000 / ($2.60 * 100) = 57.69 → 57
        assert rm.position_contracts(100_000, 2.60) == 57

    def test_too_expensive_returns_zero(self, rm: RiskManager):
        # $150 cap / ($2.00 * 100) = 0.75 → 0
        assert rm.position_contracts(1_000, 2.00) == 0

    def test_invalid_price_raises(self, rm: RiskManager):
        with pytest.raises(ValueError):
            rm.position_contracts(100_000, 0)


# ---------------------------------------------------------------------------
# should_stop_loss
# ---------------------------------------------------------------------------

class TestShouldStopLoss:
    def test_no_loss(self, rm: RiskManager):
        assert rm.should_stop_loss(1_000, 1_000, 100_000) is False

    def test_loss_below_threshold(self, rm: RiskManager):
        # 5% of $100k = $5,000; loss = $4,999
        assert rm.should_stop_loss(5_000, 1, 100_000) is False

    def test_loss_at_threshold(self, rm: RiskManager):
        # loss exactly $5,000 on $100k portfolio
        assert rm.should_stop_loss(5_000, 0, 100_000) is True

    def test_loss_above_threshold(self, rm: RiskManager):
        assert rm.should_stop_loss(10_000, 0, 100_000) is True


# ---------------------------------------------------------------------------
# should_take_profit
# ---------------------------------------------------------------------------

class TestShouldTakeProfit:
    def test_no_gain(self, rm: RiskManager):
        assert rm.should_take_profit(1_000, 1_000) is False

    def test_gain_below_threshold(self, rm: RiskManager):
        # 24.9% gain – just below 25%
        assert rm.should_take_profit(1_000, 1_249) is False

    def test_gain_at_threshold(self, rm: RiskManager):
        assert rm.should_take_profit(1_000, 1_250) is True

    def test_gain_above_threshold(self, rm: RiskManager):
        assert rm.should_take_profit(1_000, 2_000) is True

    def test_invalid_entry_raises(self, rm: RiskManager):
        with pytest.raises(ValueError):
            rm.should_take_profit(0, 100)


# ---------------------------------------------------------------------------
# should_time_exit
# ---------------------------------------------------------------------------

class TestShouldTimeExit:
    def test_fresh_position(self, rm: RiskManager):
        opened_at = datetime.now(timezone.utc)
        assert rm.should_time_exit(opened_at) is False

    def test_position_at_limit(self, rm: RiskManager):
        opened_at = datetime.now(timezone.utc) - timedelta(minutes=390)
        assert rm.should_time_exit(opened_at) is True

    def test_position_over_limit(self, rm: RiskManager):
        opened_at = datetime.now(timezone.utc) - timedelta(minutes=400)
        assert rm.should_time_exit(opened_at) is True


# ---------------------------------------------------------------------------
# exit_signal priority
# ---------------------------------------------------------------------------

class TestExitSignal:
    def test_no_trigger(self, rm: RiskManager):
        opened_at = datetime.now(timezone.utc)
        signal = rm.exit_signal(1_000, 1_100, 100_000, opened_at)
        assert signal is None

    def test_stop_loss_wins_over_time(self, rm: RiskManager):
        # Stop-loss AND time both triggered – stop_loss takes priority
        opened_at = datetime.now(timezone.utc) - timedelta(minutes=400)
        signal = rm.exit_signal(10_000, 0, 100_000, opened_at)
        assert signal == "stop_loss"

    def test_take_profit_over_time(self, rm: RiskManager):
        opened_at = datetime.now(timezone.utc) - timedelta(minutes=400)
        signal = rm.exit_signal(1_000, 1_250, 100_000, opened_at)
        assert signal == "take_profit"

    def test_time_exit_only(self, rm: RiskManager):
        opened_at = datetime.now(timezone.utc) - timedelta(minutes=400)
        signal = rm.exit_signal(1_000, 1_050, 100_000, opened_at)
        assert signal == "time_exit"
