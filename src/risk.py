"""Position sizing and risk management."""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    symbol: str           # OCC option symbol
    underlying: str
    option_type: str      # "call" or "put"
    contracts: int
    entry_price: float    # premium paid per contract (per-share basis, ×100)
    stop_loss_price: float
    take_profit_price: float
    entry_date: date = field(default_factory=date.today)

    @property
    def cost_basis(self) -> float:
        """Total cash at risk (premium paid × 100 × contracts)."""
        return self.entry_price * 100 * self.contracts


class RiskManager:
    """Enforces position sizing limits and daily loss caps."""

    def __init__(
        self,
        max_portfolio_risk_pct: float,
        max_position_risk_pct: float,
        max_open_positions: int,
        stop_loss_pct: float,
        take_profit_pct: float,
        max_daily_loss_pct: float,
    ):
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_position_risk_pct = max_position_risk_pct
        self.max_open_positions = max_open_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss_pct = max_daily_loss_pct

        self._positions: Dict[str, OpenPosition] = {}
        self._daily_realized_pnl: float = 0.0
        self._last_reset_date: date = date.today()

    # ------------------------------------------------------------------
    # Daily loss tracking
    # ------------------------------------------------------------------

    def _maybe_reset_daily_pnl(self) -> None:
        today = date.today()
        if today != self._last_reset_date:
            self._daily_realized_pnl = 0.0
            self._last_reset_date = today

    def record_trade_pnl(self, pnl: float) -> None:
        """Record realized PnL from a closed trade."""
        self._maybe_reset_daily_pnl()
        self._daily_realized_pnl += pnl
        logger.debug("Daily realized PnL updated to %.2f", self._daily_realized_pnl)

    def daily_loss_limit_reached(self, portfolio_value: float) -> bool:
        """Return True if today's realized losses exceed the daily cap."""
        self._maybe_reset_daily_pnl()
        if portfolio_value <= 0:
            return False
        daily_loss_pct = -self._daily_realized_pnl / portfolio_value
        return daily_loss_pct >= self.max_daily_loss_pct

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    @property
    def open_positions(self) -> Dict[str, OpenPosition]:
        return dict(self._positions)

    def can_open_position(self, underlying: str) -> bool:
        """Return True if a new position may be opened."""
        if len(self._positions) >= self.max_open_positions:
            logger.info(
                "Max open positions (%d) reached, cannot open new position",
                self.max_open_positions,
            )
            return False
        # One position per underlying at a time
        for pos in self._positions.values():
            if pos.underlying == underlying:
                logger.info("Already have an open position in %s", underlying)
                return False
        return True

    def size_position(
        self,
        portfolio_value: float,
        option_premium: float,
    ) -> int:
        """Return the number of contracts to buy.

        Ensures the notional risk does not exceed *max_position_risk_pct* of
        portfolio value.  Each contract controls 100 shares.
        """
        if option_premium <= 0 or portfolio_value <= 0:
            return 0
        max_risk_dollars = portfolio_value * self.max_position_risk_pct
        # We risk 100% of the premium (max loss = premium paid)
        contracts = int(max_risk_dollars / (option_premium * 100))
        return max(contracts, 0)

    def compute_stops(self, entry_price: float, option_type: str):
        """Return (stop_loss_price, take_profit_price) for an entry."""
        stop_loss = round(entry_price * (1 - self.stop_loss_pct), 4)
        take_profit = round(entry_price * (1 + self.take_profit_pct), 4)
        return stop_loss, take_profit

    def open_position(self, position: OpenPosition) -> None:
        self._positions[position.symbol] = position
        logger.info(
            "Opened position: %s (%d contracts @ %.2f)",
            position.symbol, position.contracts, position.entry_price,
        )

    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """Close the position and return realized PnL, or None if not found."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            logger.warning("Attempted to close unknown position: %s", symbol)
            return None
        pnl = (exit_price - pos.entry_price) * 100 * pos.contracts
        self.record_trade_pnl(pnl)
        logger.info(
            "Closed position: %s (PnL: %.2f)",
            symbol, pnl,
        )
        return pnl

    def positions_to_close(self, current_prices: Dict[str, float]) -> Dict[str, str]:
        """Return {symbol: reason} for positions that hit stop/profit targets."""
        to_close = {}
        for symbol, pos in self._positions.items():
            price = current_prices.get(symbol)
            if price is None:
                continue
            if price <= pos.stop_loss_price:
                to_close[symbol] = "stop_loss"
            elif price >= pos.take_profit_price:
                to_close[symbol] = "take_profit"
        return to_close
