"""
agent.py – Main agent loop, intended to be invoked every 5 minutes.

Workflow per cycle
------------------
1. Load persisted state.
2. If a position is open, check exit conditions (stop / target / max-hold).
   If triggered, log the exit and clear the position.
3. If no position, fetch 5-min candles + SMAs, detect signal.
4. On a golden or death cross, select near-ATM ≤5 DTE option, compute risk,
   store the new position.
5. Save state.

The agent does **not** send real orders; it logs what it *would* do.  Wire
up ``_enter_position`` and ``_exit_position`` to a broker SDK to go live.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data import get_candles_with_smas
from src.options import OptionContract, select_option
from src.risk import PositionRisk, RiskParams, compute_risk, should_exit
from src.signals import Signal, latest_signal
from src.state import (
    AgentState,
    PositionState,
    clear_position,
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _build_risk_params(cfg: dict) -> RiskParams:
    r = cfg.get("risk", {})
    return RiskParams(
        size_pct=r.get("size_pct", 0.15),
        stop_pct=r.get("stop_pct", 0.05),
        target_pct=r.get("target_pct", 0.25),
        max_hold_days=r.get("max_hold_days", 1),
    )


# ---------------------------------------------------------------------------
# Broker stubs (replace with real SDK calls)
# ---------------------------------------------------------------------------

def _enter_position(contract: OptionContract, risk: PositionRisk) -> None:
    """Place a BUY order for *contract*.  Stub: logs only."""
    logger.info(
        "[ORDER] BUY %d x %s %s strike=%.2f exp=%s @ %.4f  "
        "stop=%.4f target=%.4f",
        risk.contracts,
        contract.symbol,
        contract.option_type.upper(),
        contract.strike,
        contract.expiry,
        risk.entry_premium,
        risk.stop_price,
        risk.target_price,
    )


def _exit_position(pos: PositionState, reason: str) -> None:
    """Place a SELL order to close *pos*.  Stub: logs only."""
    logger.info(
        "[ORDER] SELL %d x %s %s strike=%.2f exp=%s  reason=%s",
        pos.contracts,
        pos.symbol,
        pos.option_type.upper(),
        pos.strike,
        pos.expiry,
        reason,
    )


# ---------------------------------------------------------------------------
# Current option price helper
# ---------------------------------------------------------------------------

def _get_current_premium(pos: PositionState) -> Optional[float]:
    """Return the latest mid-price for an open position."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(pos.symbol)
        chain = ticker.option_chain(pos.expiry)
        contracts = chain.calls if pos.option_type == "call" else chain.puts
        row = contracts[contracts["strike"] == pos.strike]
        if row.empty:
            return None
        bid = float(row["bid"].iloc[0] or 0)
        ask = float(row["ask"].iloc[0] or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        last = float(row["lastPrice"].iloc[0] or 0)
        return last if last > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch current premium: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core cycle logic
# ---------------------------------------------------------------------------

def _position_state_from(
    contract: OptionContract, risk: PositionRisk
) -> PositionState:
    return PositionState(
        symbol=contract.symbol,
        expiry=contract.expiry,
        strike=contract.strike,
        option_type=contract.option_type,
        entry_premium=risk.entry_premium,
        contracts=risk.contracts,
        notional=risk.notional,
        stop_price=risk.stop_price,
        target_price=risk.target_price,
        max_exit_time=risk.max_exit_time.isoformat(),
        entry_time=datetime.now(tz=timezone.utc).isoformat(),
    )


def run_cycle(state_path: str = "state.json") -> AgentState:
    """Execute one agent cycle.

    Parameters
    ----------
    state_path:
        Path to the JSON state file.

    Returns
    -------
    AgentState
        The updated state after this cycle.
    """
    cfg = _load_config()
    risk_params = _build_risk_params(cfg)
    symbols: list[str] = cfg.get("symbols", ["SPY"])
    account_equity: float = cfg.get("account_equity", 10_000.0)
    fast_period: int = cfg.get("sma_fast", 20)
    slow_period: int = cfg.get("sma_slow", 50)
    proximity_pct: float = cfg.get("proximity_pct", 1.0)
    max_dte: int = cfg.get("max_dte", 5)
    min_oi: int = cfg.get("min_open_interest", 10)

    state = load_state(state_path)
    now = datetime.now(tz=timezone.utc)

    # ------------------------------------------------------------------
    # 1. Manage open position
    # ------------------------------------------------------------------
    if state.position is not None:
        pos = state.position
        current_premium = _get_current_premium(pos)

        if current_premium is None:
            logger.warning("Cannot fetch current premium for open position; holding.")
        else:
            # Reconstruct PositionRisk for exit logic.
            p_risk = PositionRisk(
                entry_premium=pos.entry_premium,
                stop_price=pos.stop_price,
                target_price=pos.target_price,
                max_exit_time=datetime.fromisoformat(pos.max_exit_time),
                contracts=pos.contracts,
                notional=pos.notional,
            )
            exit_flag, reason = should_exit(current_premium, p_risk, now=now)
            if exit_flag:
                _exit_position(pos, reason)
                state = clear_position(state)
                logger.info("Position closed: %s", reason)
            else:
                logger.info(
                    "Position held: current=%.4f stop=%.4f target=%.4f",
                    current_premium,
                    p_risk.stop_price,
                    p_risk.target_price,
                )

    # ------------------------------------------------------------------
    # 2. Look for new entries (only when flat)
    # ------------------------------------------------------------------
    if state.position is None:
        for symbol in symbols:
            try:
                df = get_candles_with_smas(
                    symbol,
                    fast_period=fast_period,
                    slow_period=slow_period,
                )
                signal = latest_signal(
                    df,
                    fast_period=fast_period,
                    slow_period=slow_period,
                    proximity_pct=proximity_pct,
                )

                if signal == Signal.NONE:
                    logger.debug("No signal for %s", symbol)
                    state.last_signal = Signal.NONE.value
                    continue

                logger.info("Signal for %s: %s", symbol, signal.value)
                state.last_signal = signal.value

                contract = select_option(
                    symbol, signal, max_dte=max_dte, min_open_interest=min_oi
                )
                if contract is None:
                    logger.info("No suitable option found for %s; skipping.", symbol)
                    continue

                entry_premium = contract.last_price or (
                    (contract.bid + contract.ask) / 2.0
                    if contract.bid and contract.ask
                    else 0.0
                )
                if entry_premium <= 0:
                    logger.warning(
                        "Cannot determine entry premium for %s; skipping.", symbol
                    )
                    continue

                risk = compute_risk(
                    entry_premium=entry_premium,
                    account_equity=account_equity,
                    params=risk_params,
                    entry_time=now,
                )
                _enter_position(contract, risk)
                state.position = _position_state_from(contract, risk)
                # Take the first signal; skip remaining symbols.
                break

            except Exception as exc:  # noqa: BLE001
                logger.error("Error processing %s: %s", symbol, exc, exc_info=True)

    # ------------------------------------------------------------------
    # 3. Persist state
    # ------------------------------------------------------------------
    state.last_run = now.isoformat()
    state.run_count += 1
    save_state(state, path=state_path)
    logger.info("Cycle complete (run #%d)", state.run_count)
    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    run_cycle()


if __name__ == "__main__":
    main()
