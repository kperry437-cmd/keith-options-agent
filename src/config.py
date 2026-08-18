"""Configuration loading and dataclasses for the options agent."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config.json",
)


@dataclass
class RiskConfig:
    max_position_pct: float
    max_daily_loss_pct: float
    max_open_positions: int
    option_stop_loss_pct: float
    min_dte: int
    max_dte: int
    max_delta: float
    min_delta: float


@dataclass
class OptionsConfig:
    contract_quantity: int


@dataclass
class AgentConfig:
    symbol: str
    timeframe_minutes: int
    sma_periods: List[int]
    risk: RiskConfig
    options: OptionsConfig
    market_open_hour: int
    market_open_minute: int
    market_close_hour: int
    market_close_minute: int
    pre_close_buffer_minutes: int
    post_open_buffer_minutes: int


def load_config(path: str = CONFIG_PATH) -> AgentConfig:
    """Load and validate the agent configuration from *path*."""
    with open(path) as f:
        data = json.load(f)
    return AgentConfig(
        symbol=data["symbol"],
        timeframe_minutes=data["timeframe_minutes"],
        sma_periods=data["sma_periods"],
        risk=RiskConfig(**data["risk"]),
        options=OptionsConfig(**data["options"]),
        market_open_hour=data["market_open_hour"],
        market_open_minute=data["market_open_minute"],
        market_close_hour=data["market_close_hour"],
        market_close_minute=data["market_close_minute"],
        pre_close_buffer_minutes=data["pre_close_buffer_minutes"],
        post_open_buffer_minutes=data["post_open_buffer_minutes"],
    )
