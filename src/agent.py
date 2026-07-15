"""
agent.py – 5-minute SMA 20/50/200 golden/death cross options strategy.

Intended to be executed every 5 minutes by the cron schedule:
    */5 * * * *  python src/agent.py

Each run:
  1. Load configuration and persistent state.
  2. Reset daily P&L counter if a new trading day has begun.
  3. Check existing positions for stop-loss / profit-target exits.
  4. For each configured symbol, fetch the last 201 × 5-minute bars.
  5. Detect SMA golden/death crosses.
  6. Apply all risk gates (hours, daily loss, position cap, dedup).
  7. Select and purchase the best-matching options contract.
  8. Persist updated state to disk.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import pytz

# Allow `python src/agent.py` from the project root
sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from data import get_bars
from options import (
    close_contract_order,
    get_contract_price,
    get_underlying_price,
    place_market_order,
    select_contract,
)
from risk import (
    calculate_position_size,
    passes_all_checks,
    should_close_position,
)
from signals import get_signals
from state import (
    close_position,
    load_state,
    record_fill,
    reset_daily_state_if_needed,
    save_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# Need 201 bars: 200 for the slowest SMA + 1 previous bar for cross detection
_MIN_BARS = 201


def _build_clients(cfg: dict):
    """Instantiate and return (trading_client, data_client)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    paper = cfg.get("paper_trading", True)
    trading = TradingClient(
        cfg["alpaca_api_key"], cfg["alpaca_secret_key"], paper=paper
    )
    data = StockHistoricalDataClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"])
    return trading, data


def _monitor_exits(
    state: dict,
    trading_client,
    data_client,
    risk_cfg: dict,
) -> dict:
    """Check every open position for stop-loss or profit-target conditions.

    Closes qualifying positions via a market sell order and updates state.
    """
    for pos_key, pos in list(state["open_positions"].items()):
        contract_symbol = pos.get("contract", "")
        entry_price = pos.get("entry_price", 0.0)
        qty = pos.get("qty", 1)

        current_price = get_contract_price(data_client, contract_symbol)
        if current_price <= 0:
            logger.debug("No price for %s, skipping exit check", contract_symbol)
            continue

        should_exit, reason = should_close_position(
            entry_price,
            current_price,
            risk_cfg["stop_loss_pct"],
            risk_cfg["profit_target_pct"],
        )
        if should_exit:
            logger.info("Exiting %s – %s", pos_key, reason)
            try:
                close_contract_order(trading_client, contract_symbol, qty)
                state = close_position(state, pos_key, current_price)
            except Exception as exc:
                logger.error("Failed to close %s: %s", pos_key, exc)

    return state


def _trade_symbol(
    symbol: str,
    state: dict,
    trading_client,
    data_client,
    cfg: dict,
    now: datetime,
) -> dict:
    """Execute the full signal → risk-check → order flow for one *symbol*."""
    risk_cfg = cfg["risk"]
    schedule_cfg = cfg["schedule"]

    # ── 1. Fetch bars ────────────────────────────────────────────────────────
    try:
        bars = get_bars(
            symbol, _MIN_BARS, cfg["alpaca_api_key"], cfg["alpaca_secret_key"]
        )
    except Exception as exc:
        logger.error("%s: Could not fetch bars – %s", symbol, exc)
        return state

    if len(bars) < _MIN_BARS:
        logger.warning(
            "%s: Only %d bars available (need %d), skipping",
            symbol, len(bars), _MIN_BARS,
        )
        return state

    # ── 2. Detect signals ────────────────────────────────────────────────────
    signals = get_signals(bars, cfg["cross_pairs"])
    if not signals:
        logger.debug("%s: No cross detected this bar", symbol)
        return state

    # ── 3. Get underlying price ──────────────────────────────────────────────
    underlying_price = get_underlying_price(data_client, symbol)
    if underlying_price <= 0:
        # Fall back to last close in the bars DataFrame
        underlying_price = float(bars["close"].iloc[-1])

    # ── 4. Process each signal ───────────────────────────────────────────────
    try:
        account = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
    except Exception as exc:
        logger.error("Could not fetch account info: %s", exc)
        return state

    for signal in signals:
        signal_type = signal["type"]

        passed, reason = passes_all_checks(
            state,
            portfolio_value,
            symbol,
            signal_type,
            risk_cfg,
            schedule_cfg,
            now=now,
        )
        if not passed:
            logger.info("%s [%s]: Skipped – %s", symbol, signal_type, reason)
            continue

        # Contract selection
        contract = select_contract(
            trading_client,
            data_client,
            symbol,
            signal_type,
            risk_cfg,
            underlying_price,
        )
        if contract is None:
            continue

        # Determine entry price for sizing
        contract_price = get_contract_price(data_client, contract.symbol)
        if contract_price <= 0:
            # Fall back to last close on the contract if available
            contract_price = float(contract.close_price or 0)
        if contract_price <= 0:
            logger.warning(
                "%s: Cannot determine price for %s, skipping",
                symbol,
                contract.symbol,
            )
            continue

        qty = calculate_position_size(
            portfolio_value,
            risk_cfg["max_position_size_pct"],
            contract_price,
        )
        if qty <= 0:
            logger.warning(
                "%s: Position size is zero (portfolio=%.0f, price=%.2f)",
                symbol, portfolio_value, contract_price,
            )
            continue

        # Place order
        try:
            order = place_market_order(trading_client, contract.symbol, qty)
        except Exception as exc:
            logger.error("%s: Order failed – %s", symbol, exc)
            continue

        # Record position
        pos_key = f"{symbol}_{signal_type}"
        state = record_fill(
            state,
            pos_key,
            contract.symbol,
            qty,
            contract_price,
            signal,
            str(order.id),
            now=now,
        )
        logger.info(
            "Position opened: %s  contract=%s  qty=%d  price=%.2f",
            pos_key, contract.symbol, qty, contract_price,
        )

    return state


def run() -> None:
    """Entry point – executes one full agent cycle."""
    now = datetime.now(pytz.UTC)
    logger.info("=== Agent run started at %s ===", now.isoformat())

    # ── Configuration ────────────────────────────────────────────────────────
    cfg = load_config()

    # ── State ────────────────────────────────────────────────────────────────
    state = load_state()
    state = reset_daily_state_if_needed(state, now=now)

    logger.info(
        "Daily P&L: $%.2f  Open positions: %d",
        state.get("daily_pnl", 0.0),
        len(state.get("open_positions", {})),
    )

    # ── Broker clients ───────────────────────────────────────────────────────
    trading_client, data_client = _build_clients(cfg)

    # ── Exit monitoring ──────────────────────────────────────────────────────
    state = _monitor_exits(state, trading_client, data_client, cfg["risk"])

    # ── New entries ──────────────────────────────────────────────────────────
    for symbol in cfg["symbols"]:
        state = _trade_symbol(
            symbol, state, trading_client, data_client, cfg, now
        )

    # ── Persist state ────────────────────────────────────────────────────────
    save_state(state)
    logger.info("=== Agent run complete ===")


if __name__ == "__main__":
    run()
