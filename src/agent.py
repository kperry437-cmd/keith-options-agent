"""Main agent loop: 5-minute golden/death cross options agent."""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import pytz

from .config import Config, load_config
from .data import DataFetcher
from .options import OptionsChain, select_contract
from .risk import RiskManager, OpenPosition
from .signals import Signal, latest_signal
from .state import AgentState, StateMachine

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _is_market_open(cfg: Config) -> bool:
    tz = pytz.timezone(cfg.agent.timezone)
    now = datetime.now(tz=tz)
    # Skip weekends
    if now.weekday() >= 5:
        return False
    market_open = now.replace(
        hour=int(cfg.agent.market_open.split(":")[0]),
        minute=int(cfg.agent.market_open.split(":")[1]),
        second=0, microsecond=0,
    )
    market_close = now.replace(
        hour=int(cfg.agent.market_close.split(":")[0]),
        minute=int(cfg.agent.market_close.split(":")[1]),
        second=0, microsecond=0,
    )
    return market_open <= now < market_close


class OptionsAgent:
    """5-minute golden/death cross options trading agent."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._fetcher = DataFetcher(cfg.broker.api_key, cfg.broker.api_secret)
        self._chain = OptionsChain(
            cfg.broker.api_key, cfg.broker.api_secret, cfg.broker.base_url
        )
        self._risk = RiskManager(
            max_portfolio_risk_pct=cfg.risk.max_portfolio_risk_pct,
            max_position_risk_pct=cfg.risk.max_position_risk_pct,
            max_open_positions=cfg.risk.max_open_positions,
            stop_loss_pct=cfg.risk.stop_loss_pct,
            take_profit_pct=cfg.risk.take_profit_pct,
            max_daily_loss_pct=cfg.risk.max_daily_loss_pct,
        )
        # One state machine per symbol
        self._states: Dict[str, StateMachine] = {
            sym: StateMachine() for sym in cfg.symbols
        }
        self._dry_run = cfg.agent.dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocking main loop."""
        logger.info(
            "Agent started. dry_run=%s, symbols=%s", self._dry_run, self.cfg.symbols
        )
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Agent stopped by user.")
                break
            except Exception as exc:
                logger.exception("Unexpected error in main loop: %s", exc)
            time.sleep(self.cfg.agent.poll_interval_seconds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if not _is_market_open(self.cfg):
            logger.debug("Market closed — sleeping.")
            return

        portfolio_value = self._get_portfolio_value()

        # Check daily loss cap
        if self._risk.daily_loss_limit_reached(portfolio_value):
            for sm in self._states.values():
                if not sm.is_halted():
                    sm.transition(AgentState.HALTED)
            logger.warning("Daily loss limit reached. Trading halted.")
            return

        # Monitor existing positions for stop/take-profit
        self._manage_open_positions()

        # Look for new signals on each symbol
        for symbol in self.cfg.symbols:
            sm = self._states[symbol]
            if sm.is_halted():
                continue
            if not sm.is_idle():
                continue
            if not self._risk.can_open_position(symbol):
                continue

            self._evaluate_signal(symbol, portfolio_value, sm)

    def _evaluate_signal(
        self, symbol: str, portfolio_value: float, sm: StateMachine
    ) -> None:
        sig_cfg = self.cfg.signal
        try:
            bars = self._fetcher.get_bars(
                symbol,
                sig_cfg.bar_timeframe,
                sig_cfg.lookback_bars,
            )
        except Exception as exc:
            logger.error("Failed to fetch bars for %s: %s", symbol, exc)
            return

        sig = latest_signal(
            bars,
            fast_period=sig_cfg.fast_period,
            slow_period=sig_cfg.slow_period,
            ma_type=sig_cfg.ma_type,
            confirmation_bars=sig_cfg.confirmation_bars,
        )

        if sig == Signal.NONE:
            return

        logger.info("Signal detected for %s: %s", symbol, sig.value)
        sm.transition(AgentState.SIGNAL_DETECTED)

        option_type = (
            self.cfg.options.contract_type_on_golden
            if sig == Signal.GOLDEN_CROSS
            else self.cfg.options.contract_type_on_death
        )

        self._enter_trade(symbol, option_type, portfolio_value, sm)

    def _enter_trade(
        self,
        symbol: str,
        option_type: str,
        portfolio_value: float,
        sm: StateMachine,
    ) -> None:
        opt_cfg = self.cfg.options
        try:
            contracts = self._chain.get_contracts(
                symbol,
                option_type,
                opt_cfg.expiry_dte_min,
                opt_cfg.expiry_dte_max,
            )
            contract = select_contract(
                contracts,
                delta_target=opt_cfg.delta_target,
                delta_tolerance=opt_cfg.delta_tolerance,
            )
        except Exception as exc:
            logger.error("Error fetching options chain for %s: %s", symbol, exc)
            sm.reset()
            return

        if contract is None:
            logger.warning("No suitable contract found for %s %s.", symbol, option_type)
            sm.reset()
            return

        premium = contract.mid_price
        if premium is None or premium <= 0:
            logger.warning("Invalid premium for %s: %s", contract.symbol, premium)
            sm.reset()
            return

        num_contracts = self._risk.size_position(portfolio_value, premium)
        if num_contracts <= 0:
            logger.warning(
                "Position size is 0 for %s (portfolio=%.2f, premium=%.2f)",
                contract.symbol, portfolio_value, premium,
            )
            sm.reset()
            return

        stop_loss, take_profit = self._risk.compute_stops(premium, option_type)

        if self._dry_run:
            logger.info(
                "[DRY RUN] Would BUY %d x %s @ %.2f  SL=%.2f  TP=%.2f",
                num_contracts, contract.symbol, premium, stop_loss, take_profit,
            )
        else:
            self._submit_buy_order(contract.symbol, num_contracts)

        sm.transition(AgentState.ENTERING)

        position = OpenPosition(
            symbol=contract.symbol,
            underlying=symbol,
            option_type=option_type,
            contracts=num_contracts,
            entry_price=premium,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
        )
        self._risk.open_position(position)
        sm.transition(AgentState.POSITION_OPEN)

    def _manage_open_positions(self) -> None:
        current_prices = self._fetch_option_prices()
        to_close = self._risk.positions_to_close(current_prices)

        for symbol, reason in to_close.items():
            exit_price = current_prices.get(symbol, 0.0)
            pnl = self._risk.close_position(symbol, exit_price)

            # Find the state machine for this position's underlying
            for underlying, sm in self._states.items():
                for pos in []:  # placeholder; underlying tracked in OpenPosition
                    pass
            # Reset state for the underlying via the symbol lookup
            underlying = self._underlying_for_option(symbol)
            if underlying and underlying in self._states:
                self._states[underlying].reset()

            if self._dry_run:
                logger.info(
                    "[DRY RUN] Would CLOSE %s (%s) @ %.2f  PnL=%.2f",
                    symbol, reason, exit_price, pnl or 0.0,
                )
            else:
                self._submit_sell_order(symbol)

    def _underlying_for_option(self, option_symbol: str) -> Optional[str]:
        for sym in self.cfg.symbols:
            if option_symbol.startswith(sym):
                return sym
        return None

    def _fetch_option_prices(self) -> Dict[str, float]:
        """Return latest mid-prices for all open option positions."""
        prices: Dict[str, float] = {}
        for opt_symbol, pos in self._risk.open_positions.items():
            try:
                # In a real implementation, fetch live option quotes here
                # For now, return entry price (no change) as a safe default
                prices[opt_symbol] = pos.entry_price
            except Exception as exc:
                logger.error("Failed to fetch price for %s: %s", opt_symbol, exc)
        return prices

    def _get_portfolio_value(self) -> float:
        """Return the current portfolio equity."""
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(
                self.cfg.broker.api_key,
                self.cfg.broker.api_secret,
                paper=("paper" in self.cfg.broker.base_url),
            )
            account = client.get_account()
            return float(account.equity)
        except Exception as exc:
            logger.error("Could not fetch portfolio value: %s", exc)
            return 0.0

    def _submit_buy_order(self, symbol: str, contracts: int) -> None:
        """Submit a market buy order for *contracts* option contracts."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            client = TradingClient(
                self.cfg.broker.api_key,
                self.cfg.broker.api_secret,
                paper=("paper" in self.cfg.broker.base_url),
            )
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=contracts,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(order_req)
            logger.info("BUY order submitted: %d x %s", contracts, symbol)
        except Exception as exc:
            logger.error("Failed to submit buy order for %s: %s", symbol, exc)

    def _submit_sell_order(self, symbol: str) -> None:
        """Close the full position by submitting a market sell order."""
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(
                self.cfg.broker.api_key,
                self.cfg.broker.api_secret,
                paper=("paper" in self.cfg.broker.base_url),
            )
            client.close_position(symbol)
            logger.info("SELL (close) order submitted: %s", symbol)
        except Exception as exc:
            logger.error("Failed to submit sell order for %s: %s", symbol, exc)


def main() -> None:
    cfg = load_config()
    _setup_logging(cfg.agent.log_level)
    agent = OptionsAgent(cfg)
    agent.run()


if __name__ == "__main__":
    main()
