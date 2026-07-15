"""
options.py – Options contract selection and order placement via Alpaca.

Contract selection strategy:
  - Filter to the configured DTE window (min_dte … max_dte calendar days).
  - Prefer contracts whose delta is closest to ``risk_cfg['target_delta']``.
  - Fall back to picking the strike nearest to a percentage offset of the
    underlying spot price when greeks are unavailable.

Golden cross → buy CALL
Death cross  → buy PUT
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pytz

logger = logging.getLogger(__name__)

_UTC = pytz.UTC


# ---------------------------------------------------------------------------
# Contract selection
# ---------------------------------------------------------------------------


def select_contract(
    trading_client,
    data_client,
    symbol: str,
    signal_type: str,
    risk_cfg: dict,
    underlying_price: float,
) -> Optional[object]:
    """Return the best-matching options contract or None.

    Args:
        trading_client:    Alpaca TradingClient instance.
        data_client:       Alpaca OptionHistoricalDataClient instance.
        symbol:            Underlying ticker (e.g. 'SPY').
        signal_type:       ``'golden'`` (call) or ``'death'`` (put).
        risk_cfg:          Risk configuration dict.
        underlying_price:  Current spot price of *symbol*.
    """
    from alpaca.trading.enums import ContractType
    from alpaca.trading.requests import GetOptionContractsRequest

    contract_type = (
        ContractType.CALL if signal_type == "golden" else ContractType.PUT
    )

    today = datetime.now(_UTC).date()
    min_exp = today + timedelta(days=risk_cfg["min_dte"])
    max_exp = today + timedelta(days=risk_cfg["max_dte"])

    request = GetOptionContractsRequest(
        underlying_symbols=[symbol],
        status="active",
        expiration_date_gte=min_exp,
        expiration_date_lte=max_exp,
        type=contract_type,
    )

    try:
        response = trading_client.get_option_contracts(request)
    except Exception as exc:
        logger.error("Failed to fetch option contracts for %s: %s", symbol, exc)
        return None

    contracts = getattr(response, "option_contracts", [])
    if not contracts:
        logger.warning("No active %s contracts found for %s", contract_type, symbol)
        return None

    # Try greeks-based selection first
    best = _select_by_delta(contracts, risk_cfg["target_delta"], signal_type)
    if best is None:
        # Fallback: pick by strike proximity to a percentage offset from spot
        best = _select_by_strike(
            contracts, underlying_price, risk_cfg["target_delta"], signal_type
        )

    if best:
        logger.info(
            "Selected contract %s (strike=%.2f exp=%s)",
            best.symbol,
            float(best.strike_price or 0),
            best.expiration_date,
        )
    return best


def _select_by_delta(contracts: list, target_delta: float, signal_type: str):
    """Return the contract whose absolute delta is closest to *target_delta*."""
    best = None
    best_diff = float("inf")
    for c in contracts:
        if c.delta is None:
            continue
        delta = abs(float(c.delta))
        diff = abs(delta - target_delta)
        if diff < best_diff:
            best_diff = diff
            best = c
    return best


def _select_by_strike(
    contracts: list,
    spot: float,
    target_delta: float,
    signal_type: str,
) -> Optional[object]:
    """Approximate a *target_delta* strike using a percentage offset from spot.

    For calls:  target strike ≈ spot * (1 + offset)   (OTM)
    For puts:   target strike ≈ spot * (1 - offset)   (OTM)

    The offset is approximated as (1 - target_delta) * 0.1 which gives rough
    OTM percentages consistent with typical delta levels.
    """
    # Approximate OTM percentage from target delta.
    # At 0.30 delta (offset=0.07): call strike ≈ spot * 1.07, put strike ≈ spot * 0.93.
    # At 0.50 delta (offset=0.05): near-ATM strikes.
    offset = (1.0 - target_delta) * 0.10
    if signal_type == "golden":
        target_strike = spot * (1.0 + offset)
    else:
        target_strike = spot * (1.0 - offset)

    best = None
    best_diff = float("inf")
    for c in contracts:
        if c.strike_price is None:
            continue
        diff = abs(float(c.strike_price) - target_strike)
        if diff < best_diff:
            best_diff = diff
            best = c
    return best


# ---------------------------------------------------------------------------
# Price lookup
# ---------------------------------------------------------------------------


def get_contract_price(data_client, contract_symbol: str) -> float:
    """Return the latest mid-price (or last price) for *contract_symbol*.

    Returns 0.0 on any error so callers can skip gracefully.
    """
    try:
        from alpaca.data.requests import OptionLatestQuoteRequest

        req = OptionLatestQuoteRequest(symbol_or_symbols=contract_symbol)
        quote_map = data_client.get_option_latest_quote(req)
        quote = quote_map.get(contract_symbol)
        if quote and quote.ask_price and quote.bid_price:
            return (float(quote.ask_price) + float(quote.bid_price)) / 2.0
        if quote and quote.ask_price:
            return float(quote.ask_price)
    except Exception as exc:
        logger.warning("Could not fetch price for %s: %s", contract_symbol, exc)
    return 0.0


def get_underlying_price(data_client, symbol: str) -> float:
    """Return the latest trade price of the underlying equity."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trades = data_client.get_stock_latest_trade(req)
        trade = trades.get(symbol)
        if trade:
            return float(trade.price)
    except Exception as exc:
        logger.warning("Could not fetch underlying price for %s: %s", symbol, exc)
    return 0.0


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------


def place_market_order(trading_client, contract_symbol: str, qty: int) -> object:
    """Submit a day market order to buy *qty* contracts of *contract_symbol*.

    Raises on failure so the caller can log and skip to the next signal.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    request = MarketOrderRequest(
        symbol=contract_symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    order = trading_client.submit_order(request)
    logger.info("Order submitted: id=%s %dx %s", order.id, qty, contract_symbol)
    return order


def close_contract_order(
    trading_client, contract_symbol: str, qty: int
) -> object:
    """Submit a day market sell order to exit an existing position."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    request = MarketOrderRequest(
        symbol=contract_symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    order = trading_client.submit_order(request)
    logger.info("Close order submitted: id=%s %dx %s", order.id, qty, contract_symbol)
    return order
