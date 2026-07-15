# keith-options-agent

An automated options trading agent with configurable risk management.

## Risk parameters

Defined in `config.json`:

| Parameter | Value | Meaning |
|---|---|---|
| `max_allocation_pct` | 0.15 | Maximum 15 % of portfolio equity per position |
| `max_loss_pct` | 0.05 | Stop-loss at 5 % of portfolio equity |
| `take_profit_pct` | 0.25 | Take-profit at 25 % gain on entry cost |
| `max_hold_minutes` | 390 | Force-exit after 390 minutes (one full trading session) |

## Project layout

```
.
├── config.json            # Risk parameters (loaded at runtime)
├── requirements.txt
├── src/
│   ├── config.py          # AgentConfig dataclass + JSON loader
│   ├── risk_manager.py    # RiskManager – position sizing & exit signals
│   ├── position_manager.py# PositionManager – tracks open positions
│   └── agent.py           # OptionsAgent orchestrator + Broker ABC
└── tests/
    └── test_risk_manager.py
```

## Getting started

```bash
pip install -r requirements.txt
```

Implement the `Broker` abstract class in `src/agent.py` to connect a live or
paper-trading broker (e.g. Alpaca), inject a scanner callable that returns
`OptionCandidate` objects, then call `agent.run_cycle()` on your desired schedule:

```python
from src.agent import OptionsAgent
from src.config import AgentConfig

agent = OptionsAgent(
    broker=MyBroker(),
    config=AgentConfig.from_file(),   # reads config.json
    scanner=my_scanner,
)

# Call periodically (e.g. every minute during market hours)
agent.run_cycle()
```

## Running tests

```bash
python -m pytest tests/ -v
```
