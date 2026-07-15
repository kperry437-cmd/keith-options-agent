"""Configuration loader for the options agent."""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    name: str
    base_url: str
    data_url: str
    api_key: str
    api_secret: str


@dataclass
class SignalConfig:
    fast_period: int
    slow_period: int
    ma_type: str
    bar_timeframe: str
    lookback_bars: int
    confirmation_bars: int


@dataclass
class OptionsConfig:
    expiry_dte_min: int
    expiry_dte_max: int
    delta_target: float
    delta_tolerance: float
    contract_type_on_golden: str
    contract_type_on_death: str


@dataclass
class RiskConfig:
    max_portfolio_risk_pct: float
    max_position_risk_pct: float
    max_open_positions: int
    stop_loss_pct: float
    take_profit_pct: float
    max_daily_loss_pct: float


@dataclass
class AgentConfig:
    poll_interval_seconds: int
    market_open: str
    market_close: str
    timezone: str
    dry_run: bool
    log_level: str


@dataclass
class Config:
    broker: BrokerConfig
    symbols: List[str]
    signal: SignalConfig
    options: OptionsConfig
    risk: RiskConfig
    agent: AgentConfig


def load_config(path: str = "config.json") -> Config:
    """Load configuration from a JSON file, with env-var overrides for secrets."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = json.load(f)

    broker_raw = raw["broker"]
    # Allow environment variable overrides for credentials
    api_key = os.environ.get("ALPACA_API_KEY", broker_raw.get("api_key", ""))
    api_secret = os.environ.get("ALPACA_API_SECRET", broker_raw.get("api_secret", ""))

    broker = BrokerConfig(
        name=broker_raw["name"],
        base_url=broker_raw["base_url"],
        data_url=broker_raw["data_url"],
        api_key=api_key,
        api_secret=api_secret,
    )

    sig_raw = raw["signal"]
    signal = SignalConfig(
        fast_period=sig_raw["fast_period"],
        slow_period=sig_raw["slow_period"],
        ma_type=sig_raw["ma_type"],
        bar_timeframe=sig_raw["bar_timeframe"],
        lookback_bars=sig_raw["lookback_bars"],
        confirmation_bars=sig_raw["confirmation_bars"],
    )

    opt_raw = raw["options"]
    options = OptionsConfig(
        expiry_dte_min=opt_raw["expiry_dte_min"],
        expiry_dte_max=opt_raw["expiry_dte_max"],
        delta_target=opt_raw["delta_target"],
        delta_tolerance=opt_raw["delta_tolerance"],
        contract_type_on_golden=opt_raw["contract_type_on_golden"],
        contract_type_on_death=opt_raw["contract_type_on_death"],
    )

    risk_raw = raw["risk"]
    risk = RiskConfig(
        max_portfolio_risk_pct=risk_raw["max_portfolio_risk_pct"],
        max_position_risk_pct=risk_raw["max_position_risk_pct"],
        max_open_positions=risk_raw["max_open_positions"],
        stop_loss_pct=risk_raw["stop_loss_pct"],
        take_profit_pct=risk_raw["take_profit_pct"],
        max_daily_loss_pct=risk_raw["max_daily_loss_pct"],
    )

    agent_raw = raw["agent"]
    agent = AgentConfig(
        poll_interval_seconds=agent_raw["poll_interval_seconds"],
        market_open=agent_raw["market_open"],
        market_close=agent_raw["market_close"],
        timezone=agent_raw["timezone"],
        dry_run=agent_raw["dry_run"],
        log_level=agent_raw["log_level"],
    )

    cfg = Config(
        broker=broker,
        symbols=raw["symbols"],
        signal=signal,
        options=options,
        risk=risk,
        agent=agent,
    )
    logger.debug("Configuration loaded from %s", path)
    return cfg
