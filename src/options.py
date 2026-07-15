"""Options chain querying and contract selection."""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType, ExerciseStyle
    _ALPACA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ALPACA_AVAILABLE = False


@dataclass
class OptionContract:
    symbol: str           # OCC symbol, e.g. "SPY240119C00480000"
    underlying: str
    expiry: date
    strike: float
    option_type: str      # "call" or "put"
    delta: Optional[float]
    ask: Optional[float]
    bid: Optional[float]
    open_interest: Optional[int]

    @property
    def mid_price(self) -> Optional[float]:
        if self.ask is not None and self.bid is not None:
            return round((self.ask + self.bid) / 2, 2)
        return None

    @property
    def dte(self) -> int:
        return (self.expiry - date.today()).days


class OptionsChain:
    """Fetches and filters an options chain for a given underlying."""

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        if not _ALPACA_AVAILABLE:
            raise ImportError("alpaca-py is required. Install with: pip install alpaca-py")
        self._client = TradingClient(api_key, api_secret, paper=("paper" in base_url))

    def get_contracts(
        self,
        underlying: str,
        option_type: str,
        dte_min: int,
        dte_max: int,
    ) -> List[OptionContract]:
        """Return available contracts filtered by type and expiry window."""
        today = date.today()
        expiry_start = today + timedelta(days=dte_min)
        expiry_end = today + timedelta(days=dte_max)

        ct = ContractType.CALL if option_type == "call" else ContractType.PUT

        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            type=ct,
            expiration_date_gte=expiry_start,
            expiration_date_lte=expiry_end,
        )

        response = self._client.get_option_contracts(req)
        contracts = []
        for c in response.option_contracts:
            contracts.append(
                OptionContract(
                    symbol=c.symbol,
                    underlying=underlying,
                    expiry=c.expiration_date,
                    strike=float(c.strike_price),
                    option_type=option_type,
                    delta=float(c.delta) if c.delta is not None else None,
                    ask=float(c.ask_price) if c.ask_price is not None else None,
                    bid=float(c.bid_price) if c.bid_price is not None else None,
                    open_interest=int(c.open_interest) if c.open_interest is not None else None,
                )
            )
        logger.debug(
            "Found %d %s contracts for %s (DTE %d–%d)",
            len(contracts), option_type, underlying, dte_min, dte_max,
        )
        return contracts


def select_contract(
    contracts: List[OptionContract],
    delta_target: float,
    delta_tolerance: float,
) -> Optional[OptionContract]:
    """Select the contract whose delta is closest to *delta_target*.

    Only considers contracts within *delta_tolerance* of the target.
    Prefers higher open interest among ties.
    """
    if not contracts:
        return None

    candidates = [
        c for c in contracts
        if c.delta is not None
        and abs(abs(c.delta) - delta_target) <= delta_tolerance
        and c.ask is not None
        and c.ask > 0
    ]

    if not candidates:
        logger.warning(
            "No contracts within delta target %.2f ± %.2f", delta_target, delta_tolerance
        )
        return None

    # Sort by closeness to target delta, then by open interest descending
    candidates.sort(
        key=lambda c: (abs(abs(c.delta) - delta_target), -(c.open_interest or 0))
    )
    return candidates[0]
