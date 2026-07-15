"""Options trading agent orchestrator.

The agent follows a simple loop:

1. Fetch the current portfolio equity from the broker.
2. Scan for candidate option trades via ``scan_opportunities``.
3. For each candidate, check risk limits (allocation cap) before placing an order.
4. Monitor all open positions and exit those that hit a risk trigger.

Broker integration is deliberately kept behind an abstract ``Broker`` interface so
the agent can be wired to any back-end (Alpaca, paper trading, back-testing, etc.)
without changing core logic.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional

from .config import AgentConfig
from .position_manager import Position, PositionManager
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract broker interface
# ---------------------------------------------------------------------------


class Broker(ABC):
    """Minimal interface that a broker implementation must satisfy."""

    @abstractmethod
    def get_portfolio_equity(self) -> float:
        """Return current total equity in dollars."""

    @abstractmethod
    def get_option_price(self, position_id: str) -> float:
        """Return the current per-share mid-price for an open position."""

    @abstractmethod
    def place_option_order(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiration: str,
        contracts: int,
        limit_price: float,
    ) -> bool:
        """Submit a buy order for *contracts* contracts at *limit_price*.

        Returns True when the order is accepted, False otherwise.
        """

    @abstractmethod
    def close_position(self, position_id: str) -> bool:
        """Submit a market order to close the position identified by *position_id*.

        Returns True when the order is accepted, False otherwise.
        """


# ---------------------------------------------------------------------------
# Candidate trade dataclass
# ---------------------------------------------------------------------------


class OptionCandidate:
    """A potential option trade surfaced by the scanning step."""

    def __init__(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiration: str,
        ask_price: float,
    ) -> None:
        self.symbol = symbol
        self.option_type = option_type
        self.strike = strike
        self.expiration = expiration
        self.ask_price = ask_price


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class OptionsAgent:
    """Core options trading agent.

    Args:
        broker: Concrete broker implementation.
        config: Risk and execution configuration.
        scanner: Callable that returns a list of OptionCandidate objects.
            Injected so scanning logic can be swapped without modifying the agent.
    """

    def __init__(
        self,
        broker: Broker,
        config: Optional[AgentConfig] = None,
        scanner=None,
    ) -> None:
        self._broker = broker
        self._cfg = config or AgentConfig.from_file()
        self._risk = RiskManager(self._cfg)
        self._positions = PositionManager()
        self._scanner = scanner or self._default_scanner

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_cycle(self) -> None:
        """Execute one iteration of the scan → enter → monitor loop."""
        equity = self._broker.get_portfolio_equity()
        logger.info("Portfolio equity: $%.2f", equity)

        self._monitor_positions(equity)
        self._enter_positions(equity)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def _monitor_positions(self, equity: float) -> None:
        """Check exit conditions for every open position."""
        to_close: list[str] = []

        for pid, pos in self._positions.all():
            try:
                current_price = self._broker.get_option_price(pid)
            except Exception as exc:
                logger.warning("Could not fetch price for %s: %s", pid, exc)
                continue

            current_value = pos.current_value(current_price)
            reason = self._risk.exit_signal(
                entry_cost=pos.entry_cost,
                current_value=current_value,
                portfolio_equity=equity,
                opened_at=pos.opened_at,
            )
            if reason:
                logger.info(
                    "Exit signal %r for %s (P&L: $%.2f)",
                    reason,
                    pid,
                    pos.unrealised_pnl(current_price),
                )
                to_close.append(pid)

        for pid in to_close:
            self._close_position(pid)

    def _close_position(self, position_id: str) -> None:
        success = self._broker.close_position(position_id)
        if success:
            self._positions.remove(position_id)
            logger.info("Closed position %s", position_id)
        else:
            logger.error("Failed to close position %s", position_id)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def _enter_positions(self, equity: float) -> None:
        """Scan for candidates and open positions within risk limits."""
        candidates: List[OptionCandidate] = self._scanner()
        for candidate in candidates:
            self._try_enter(candidate, equity)

    def _try_enter(self, candidate: OptionCandidate, equity: float) -> None:
        """Attempt to open a position for *candidate* if risk rules allow."""
        contracts = self._risk.position_contracts(
            portfolio_equity=equity,
            option_price=candidate.ask_price,
        )
        if contracts == 0:
            logger.debug(
                "Skipping %s %s: 0 contracts fit within allocation cap",
                candidate.symbol,
                candidate.option_type,
            )
            return

        # Build the prospective position to derive its id before placing the order.
        pos = Position(
            symbol=candidate.symbol,
            option_type=candidate.option_type,
            strike=candidate.strike,
            expiration=candidate.expiration,
            contracts=contracts,
            entry_price=candidate.ask_price,
            opened_at=datetime.now(timezone.utc),
        )
        pid = PositionManager.make_position_id(pos)

        if pid in self._positions:
            logger.debug("Already holding %s – skipping", pid)
            return

        success = self._broker.place_option_order(
            symbol=candidate.symbol,
            option_type=candidate.option_type,
            strike=candidate.strike,
            expiration=candidate.expiration,
            contracts=contracts,
            limit_price=candidate.ask_price,
        )
        if success:
            self._positions.add(pos)
            logger.info(
                "Opened %d contract(s) of %s (cost: $%.2f)",
                contracts,
                pid,
                pos.entry_cost,
            )
        else:
            logger.warning("Order rejected for %s", pid)

    # ------------------------------------------------------------------
    # Default no-op scanner
    # ------------------------------------------------------------------

    @staticmethod
    def _default_scanner() -> List[OptionCandidate]:
        """Placeholder scanner – returns no candidates.

        Replace by passing a real ``scanner`` callable to the constructor.
        """
        return []
