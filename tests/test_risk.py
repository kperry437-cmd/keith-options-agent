"""
test_risk.py – Unit tests for risk management checks and helpers.
"""

from datetime import datetime, timedelta

import pytz
import pytest

from risk import (
    calculate_position_size,
    check_daily_loss,
    check_duplicate_position,
    check_position_count,
    is_market_hours,
    passes_all_checks,
    should_close_position,
)

_ET = pytz.timezone("America/New_York")
_UTC = pytz.UTC


# ---------------------------------------------------------------------------
# is_market_hours
# ---------------------------------------------------------------------------


def _et(hour: int, minute: int) -> datetime:
    """Return a UTC datetime corresponding to the given ET time today."""
    now_et = datetime.now(_ET).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return now_et.astimezone(_UTC)


class TestIsMarketHours:
    def test_within_window(self):
        # 11:00 ET is comfortably inside the default window (10:00 – 15:30)
        assert is_market_hours(_et(11, 0)) is True

    def test_too_early(self):
        # 9:40 ET is within the open buffer (30 min after 9:30 = 10:00)
        assert is_market_hours(_et(9, 40)) is False

    def test_too_late(self):
        # 15:45 ET is within the close buffer (30 min before 16:00 = 15:30)
        assert is_market_hours(_et(15, 45)) is False

    def test_exactly_at_trade_start(self):
        # 10:00 ET with default 30-min buffer → trade_start = 10:00
        assert is_market_hours(_et(10, 0)) is True

    def test_exactly_at_trade_end(self):
        # 15:30 ET with default 30-min buffer → trade_end = 15:30
        assert is_market_hours(_et(15, 30)) is True

    def test_before_open(self):
        assert is_market_hours(_et(8, 0)) is False

    def test_after_close(self):
        assert is_market_hours(_et(17, 0)) is False

    def test_custom_buffers(self):
        # With zero buffer: full 9:30–16:00 window
        assert is_market_hours(_et(9, 30), open_buffer_min=0, close_buffer_min=0) is True
        assert is_market_hours(_et(16, 0), open_buffer_min=0, close_buffer_min=0) is True


# ---------------------------------------------------------------------------
# check_daily_loss
# ---------------------------------------------------------------------------


class TestCheckDailyLoss:
    def test_no_loss_passes(self):
        assert check_daily_loss(0.0, 100_000, 0.05) is True

    def test_profit_passes(self):
        assert check_daily_loss(500.0, 100_000, 0.05) is True

    def test_small_loss_passes(self):
        assert check_daily_loss(-2_000.0, 100_000, 0.05) is True  # 2% < 5%

    def test_loss_at_limit_fails(self):
        assert check_daily_loss(-5_000.0, 100_000, 0.05) is False  # exactly 5%

    def test_loss_over_limit_fails(self):
        assert check_daily_loss(-6_000.0, 100_000, 0.05) is False

    def test_zero_portfolio_fails(self):
        assert check_daily_loss(0.0, 0, 0.05) is False


# ---------------------------------------------------------------------------
# check_position_count
# ---------------------------------------------------------------------------


class TestCheckPositionCount:
    def test_empty_passes(self):
        assert check_position_count({}, 5) is True

    def test_below_limit_passes(self):
        assert check_position_count({"a": 1, "b": 2}, 5) is True

    def test_at_limit_fails(self):
        positions = {str(i): i for i in range(5)}
        assert check_position_count(positions, 5) is False

    def test_over_limit_fails(self):
        positions = {str(i): i for i in range(6)}
        assert check_position_count(positions, 5) is False


# ---------------------------------------------------------------------------
# check_duplicate_position
# ---------------------------------------------------------------------------


class TestCheckDuplicatePosition:
    def test_no_existing_passes(self):
        assert check_duplicate_position({}, "SPY", "golden") is True

    def test_different_symbol_passes(self):
        assert check_duplicate_position({"QQQ_golden": {}}, "SPY", "golden") is True

    def test_different_direction_passes(self):
        assert check_duplicate_position({"SPY_death": {}}, "SPY", "golden") is True

    def test_exact_duplicate_fails(self):
        assert check_duplicate_position({"SPY_golden": {}}, "SPY", "golden") is False


# ---------------------------------------------------------------------------
# calculate_position_size
# ---------------------------------------------------------------------------


class TestCalculatePositionSize:
    def test_basic_sizing(self):
        # 2% of $100k = $2000 / (5.00 * 100) = 4 contracts
        assert calculate_position_size(100_000, 0.02, 5.00) == 4

    def test_minimum_one_contract(self):
        # Very expensive contract still yields at least 1
        assert calculate_position_size(10_000, 0.01, 500.0) == 1

    def test_zero_price_returns_zero(self):
        assert calculate_position_size(100_000, 0.02, 0.0) == 0

    def test_zero_portfolio_returns_zero(self):
        assert calculate_position_size(0, 0.02, 5.0) == 0

    def test_multiple_contracts(self):
        # 5% of $200k = $10k / (2.50 * 100) = 40 contracts
        assert calculate_position_size(200_000, 0.05, 2.50) == 40


# ---------------------------------------------------------------------------
# should_close_position
# ---------------------------------------------------------------------------


class TestShouldClosePosition:
    def test_no_close_in_range(self):
        closed, reason = should_close_position(1.00, 1.20, 0.50, 1.00)
        assert closed is False
        assert reason == ""

    def test_stop_loss_triggered(self):
        # Entry $1.00, now $0.40 → -60% loss, limit is -50%
        closed, reason = should_close_position(1.00, 0.40, 0.50, 1.00)
        assert closed is True
        assert "Stop-loss" in reason

    def test_stop_loss_exactly_at_limit(self):
        closed, _ = should_close_position(1.00, 0.50, 0.50, 1.00)
        assert closed is True

    def test_profit_target_triggered(self):
        # Entry $1.00, now $2.10 → +110% gain, target is +100%
        closed, reason = should_close_position(1.00, 2.10, 0.50, 1.00)
        assert closed is True
        assert "Profit target" in reason

    def test_profit_target_exactly_at_limit(self):
        closed, _ = should_close_position(1.00, 2.00, 0.50, 1.00)
        assert closed is True

    def test_zero_entry_price_safe(self):
        closed, _ = should_close_position(0.0, 1.0, 0.50, 1.00)
        assert closed is False


# ---------------------------------------------------------------------------
# passes_all_checks (integration of individual checks)
# ---------------------------------------------------------------------------


def _make_state(daily_pnl: float = 0.0, open_positions: dict | None = None) -> dict:
    return {
        "daily_pnl": daily_pnl,
        "open_positions": open_positions or {},
    }


def _valid_now() -> datetime:
    """Return a datetime that is comfortably inside the trading window."""
    now_et = datetime.now(_ET).replace(
        hour=11, minute=0, second=0, microsecond=0
    )
    return now_et.astimezone(_UTC)


_RISK = {
    "max_daily_loss_pct": 0.05,
    "max_open_positions": 5,
}
_SCHEDULE = {
    "market_open_buffer_min": 30,
    "market_close_buffer_min": 30,
}


class TestPassesAllChecks:
    def test_all_pass(self):
        passed, reason = passes_all_checks(
            _make_state(), 100_000, "SPY", "golden", _RISK, _SCHEDULE, now=_valid_now()
        )
        assert passed is True
        assert reason == "OK"

    def test_fails_outside_hours(self):
        off_hours = datetime.now(_ET).replace(
            hour=7, minute=0, second=0, microsecond=0
        ).astimezone(_UTC)
        passed, reason = passes_all_checks(
            _make_state(), 100_000, "SPY", "golden", _RISK, _SCHEDULE, now=off_hours
        )
        assert passed is False
        assert "trading hours" in reason.lower()

    def test_fails_daily_loss(self):
        state = _make_state(daily_pnl=-6_000.0)
        passed, reason = passes_all_checks(
            state, 100_000, "SPY", "golden", _RISK, _SCHEDULE, now=_valid_now()
        )
        assert passed is False
        assert "loss" in reason.lower()

    def test_fails_max_positions(self):
        positions = {str(i): {} for i in range(5)}
        state = _make_state(open_positions=positions)
        passed, reason = passes_all_checks(
            state, 100_000, "SPY", "golden", _RISK, _SCHEDULE, now=_valid_now()
        )
        assert passed is False
        assert "position" in reason.lower()

    def test_fails_duplicate(self):
        state = _make_state(open_positions={"SPY_golden": {}})
        passed, reason = passes_all_checks(
            state, 100_000, "SPY", "golden", _RISK, _SCHEDULE, now=_valid_now()
        )
        assert passed is False
        assert "SPY" in reason
