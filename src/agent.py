"""
keith-5m-options-cross-agent
============================
5-minute SMA 20/50/200 golden/death cross options strategy.

Runs on a 5-minute schedule (configured in the agent YAML).
Environment variables required:
  ALPACA_API_KEY    — Alpaca API key ID
  ALPACA_SECRET_KEY — Alpaca secret key
  ALPACA_PAPER      — "true" (default) to use the paper-trading endpoint
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Tuple

import pandas as pd
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderSide, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest
from dotenv import load_dotenv

from .config import load_config
from .position_manager import Position, PositionManager
from .risk_manager import RiskManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

class Signal(Enum):
    GOLDEN_CROSS_FAST = "golden_cross_fast"  # SMA20 crosses above SMA50
    GOLDEN_CROSS_SLOW = "golden_cross_slow"  # SMA50 crosses above SMA200
    DEATH_CROSS_FAST = "death_cross_fast"    # SMA20 crosses below SMA50
    DEATH_CROSS_SLOW = "death_cross_slow"    # SMA50 crosses below SMA200
    NONE = "none"


def _detect_cross(fast: pd.Series, slow: pd.Series) -> Optional[str]:
    """Return ``'golden'``, ``'death'``, or ``None`` for the last two bars."""
    if len(fast) < 2 or len(slow) < 2:
        return None
    prev_f, curr_f = fast.iloc[-2], fast.iloc[-1]
    prev_s, curr_s = slow.iloc[-2], slow.iloc[-1]
    if any(pd.isna(v) for v in (prev_f, curr_f, prev_s, curr_s)):
        return None
    if prev_f <= prev_s and curr_f > curr_s:
        return "golden"
    if prev_f >= prev_s and curr_f < curr_s:
        return "death"
    return None


def get_signal(closes: pd.Series) -> Signal:
    """Evaluate SMA 20/50/200 crosses on *closes*.

    Priority: slow cross (SMA50/200) supersedes fast cross (SMA20/50).
    """
    sma20 = closes.rolling(20).mean()
    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()

    slow = _detect_cross(sma50, sma200)
    if slow == "golden":
        return Signal.GOLDEN_CROSS_SLOW
    if slow == "death":
        return Signal.DEATH_CROSS_SLOW

    fast = _detect_cross(sma20, sma50)
    if fast == "golden":
        return Signal.GOLDEN_CROSS_FAST
    if fast == "death":
        return Signal.DEATH_CROSS_FAST

    return Signal.NONE


# ---------------------------------------------------------------------------
# Option contract selection
# ---------------------------------------------------------------------------

def _select_option_contract(
    trading_client: TradingClient,
    option_client: OptionHistoricalDataClient,
    underlying: str,
    contract_type: ContractType,
    min_dte: int,
    max_dte: int,
    min_delta: float,
    max_delta: float,
) -> Optional[Tuple[str, float, float, int]]:
    """Return ``(symbol, price, delta, dte)`` for the best matching contract.

    "Best" is defined as the contract whose |delta| is closest to the midpoint
    of [min_delta, max_delta].  Returns ``None`` if no suitable contract exists.
    """
    today = date.today()
    exp_min = today + timedelta(days=min_dte)
    exp_max = today + timedelta(days=max_dte)

    try:
        resp = trading_client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[underlying],
                type=contract_type,
                expiration_date_gte=str(exp_min),
                expiration_date_lte=str(exp_max),
            )
        )
    except Exception as exc:
        logger.error("Failed to fetch option contracts: %s", exc)
        return None

    contracts = resp.option_contracts
    if not contracts:
        logger.warning("No option contracts found for %s", underlying)
        return None

    symbols = [c.symbol for c in contracts]
    try:
        snapshots = option_client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=symbols)
        )
    except Exception as exc:
        logger.error("Failed to fetch option snapshots: %s", exc)
        return None

    target_delta = (min_delta + max_delta) / 2
    best_symbol: Optional[str] = None
    best_delta_diff = float("inf")
    best_price = 0.0
    best_delta = 0.0
    best_dte = 0

    for contract in contracts:
        snap = snapshots.get(contract.symbol)
        if snap is None or snap.greeks is None:
            continue
        delta = abs(snap.greeks.delta or 0.0)
        if not (min_delta <= delta <= max_delta):
            continue

        # Prefer ask price; fall back to last trade price
        price: float = 0.0
        if snap.latest_quote and snap.latest_quote.ask_price:
            price = float(snap.latest_quote.ask_price)
        elif snap.latest_trade and snap.latest_trade.price:
            price = float(snap.latest_trade.price)
        if price <= 0:
            continue

        dte = (contract.expiration_date - today).days
        delta_diff = abs(delta - target_delta)
        if delta_diff < best_delta_diff:
            best_delta_diff = delta_diff
            best_symbol = contract.symbol
            best_price = price
            best_delta = delta
            best_dte = dte

    if best_symbol is None:
        logger.warning(
            "No %s contract met delta [%.2f, %.2f] for %s",
            contract_type.value,
            min_delta,
            max_delta,
            underlying,
        )
        return None

    return best_symbol, best_price, best_delta, best_dte


# ---------------------------------------------------------------------------
# Stop-loss monitor
# ---------------------------------------------------------------------------

def _check_stop_losses(
    position_manager: PositionManager,
    risk_manager: RiskManager,
    option_client: OptionHistoricalDataClient,
    trading_client: TradingClient,
) -> None:
    """Close any open positions that have breached the stop-loss threshold."""
    for pos in list(position_manager.open_positions):
        try:
            snaps = option_client.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=[pos.symbol])
            )
            snap = snaps.get(pos.symbol)
        except Exception as exc:
            logger.error("Snapshot fetch failed for %s: %s", pos.symbol, exc)
            continue

        if snap is None:
            continue

        current_price: float = 0.0
        if snap.latest_trade and snap.latest_trade.price:
            current_price = float(snap.latest_trade.price)
        elif snap.latest_quote and snap.latest_quote.bid_price:
            current_price = float(snap.latest_quote.bid_price)
        if current_price <= 0:
            continue

        position_manager.update_price(pos.symbol, current_price)

        if not risk_manager.should_stop_loss(pos.entry_price, current_price):
            continue

        logger.info(
            "Stop-loss triggered — %s: current %.2f vs entry %.2f",
            pos.symbol,
            current_price,
            pos.entry_price,
        )
        try:
            trading_client.submit_order(
                MarketOrderRequest(
                    symbol=pos.symbol,
                    qty=pos.contracts,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            pnl = position_manager.close_position(pos.symbol, current_price)
            logger.info("Closed %s — realized P&L: $%.2f", pos.symbol, pnl)
        except Exception as exc:
            logger.error("Failed to submit stop-loss order for %s: %s", pos.symbol, exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent() -> None:
    """Execute one iteration of the cross-strategy agent."""
    config = load_config()

    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

    trading_client = TradingClient(api_key, secret_key, paper=paper)
    stock_client = StockHistoricalDataClient(api_key, secret_key)
    option_client = OptionHistoricalDataClient(api_key, secret_key)

    risk_manager = RiskManager(config)
    position_manager = PositionManager()

    # Use Eastern Time so market-hour comparisons are accurate
    now_et = datetime.now(ET)

    # ------------------------------------------------------------------
    # 1. Monitor and close any stop-loss breaches
    # ------------------------------------------------------------------
    _check_stop_losses(position_manager, risk_manager, option_client, trading_client)

    # ------------------------------------------------------------------
    # 2. Fetch account equity
    # ------------------------------------------------------------------
    try:
        account = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
    except Exception as exc:
        logger.error("Failed to fetch account info: %s", exc)
        return

    # ------------------------------------------------------------------
    # 3. Fetch recent 5-minute bars and compute the signal
    # ------------------------------------------------------------------
    bars_needed = max(config.sma_periods) + 10  # small buffer above 200
    # 200 five-minute bars span ~4 trading days; fetch 30 calendar days to be safe
    start_utc = datetime.now(timezone.utc) - timedelta(days=30)
    end_utc = datetime.now(timezone.utc)

    try:
        raw = stock_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=config.symbol,
                timeframe=TimeFrame(config.timeframe_minutes, TimeFrameUnit.Minute),
                start=start_utc,
                end=end_utc,
                limit=bars_needed,
            )
        )
        bars_df = raw.df
    except Exception as exc:
        logger.error("Failed to fetch bars for %s: %s", config.symbol, exc)
        return

    if isinstance(bars_df.index, pd.MultiIndex):
        bars_df = bars_df.loc[config.symbol]

    required_bars = max(config.sma_periods)
    if len(bars_df) < required_bars:
        logger.warning(
            "Insufficient bars: need %d, got %d", required_bars, len(bars_df)
        )
        return

    closes = bars_df["close"]
    signal = get_signal(closes)
    logger.info(
        "Signal: %s | last close: %.4f | bars: %d",
        signal.value,
        closes.iloc[-1],
        len(closes),
    )

    if signal is Signal.NONE:
        return

    # ------------------------------------------------------------------
    # 4. Determine trade direction
    # ------------------------------------------------------------------
    if signal in (Signal.GOLDEN_CROSS_FAST, Signal.GOLDEN_CROSS_SLOW):
        contract_type = ContractType.CALL
    else:
        contract_type = ContractType.PUT

    # ------------------------------------------------------------------
    # 5. Select the best option contract
    # ------------------------------------------------------------------
    result = _select_option_contract(
        trading_client=trading_client,
        option_client=option_client,
        underlying=config.symbol,
        contract_type=contract_type,
        min_dte=config.risk.min_dte,
        max_dte=config.risk.max_dte,
        min_delta=config.risk.min_delta,
        max_delta=config.risk.max_delta,
    )
    if result is None:
        return
    opt_symbol, opt_price, opt_delta, opt_dte = result

    # ------------------------------------------------------------------
    # 6. Pre-trade risk approval
    # ------------------------------------------------------------------
    approval = risk_manager.approve_trade(
        now=now_et,
        daily_pnl=position_manager.daily_pnl,
        portfolio_value=portfolio_value,
        open_position_count=position_manager.open_position_count,
        option_delta=opt_delta,
        dte=opt_dte,
    )
    if not approval.allowed:
        logger.info("Trade blocked by risk manager: %s", approval.reason)
        return

    # ------------------------------------------------------------------
    # 7. Size the position and submit the order
    # ------------------------------------------------------------------
    size = risk_manager.calculate_position_size(portfolio_value, opt_price)
    qty = min(size.contracts, config.options.contract_quantity)
    if qty < 1:
        logger.warning("Position size is zero — skipping order")
        return

    try:
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=opt_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        logger.info(
            "Order submitted — %s x%d | signal: %s | delta: %.2f | DTE: %d",
            opt_symbol,
            qty,
            signal.value,
            opt_delta,
            opt_dte,
        )
        position_manager.open_position(
            Position(
                symbol=opt_symbol,
                underlying=config.symbol,
                side=contract_type.value.lower(),
                contracts=qty,
                entry_price=opt_price,
                entry_time=now_et.isoformat(),
                current_price=opt_price,
            )
        )
    except Exception as exc:
        logger.error("Failed to submit order for %s: %s", opt_symbol, exc)


if __name__ == "__main__":
    run_agent()
