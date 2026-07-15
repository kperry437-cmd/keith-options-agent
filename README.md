# keith-options-agent

A Python-based algorithmic options trading agent that detects **golden cross** and **death cross** signals on 5-minute bars and automatically buys the corresponding options contract.

---

## How it works

| Signal | Condition | Action |
|---|---|---|
| **Golden Cross** | Fast MA crosses *above* slow MA | Buy a **call** option |
| **Death Cross** | Fast MA crosses *below* slow MA | Buy a **put** option |

Default moving averages: **9-period EMA** (fast) and **21-period EMA** (slow) on 5-minute bars.

---

## Project structure

```
keith-options-agent/
├── config.json          # All tunable parameters
├── requirements.txt
├── src/
│   ├── config.py        # Configuration loader
│   ├── data.py          # 5-min OHLCV bar fetcher (Alpaca)
│   ├── signals.py       # Golden/death cross detection
│   ├── options.py       # Options chain querying & contract selection
│   ├── risk.py          # Position sizing & daily loss management
│   ├── state.py         # Trade state machine
│   └── agent.py         # Main agent loop
└── tests/
    ├── test_signals.py  # Signal detection unit tests
    └── test_risk.py     # Risk management unit tests
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set your Alpaca credentials as environment variables (do **not** hard-code them):

```bash
export ALPACA_API_KEY="your-key"
export ALPACA_API_SECRET="your-secret"
```

---

## Configuration (`config.json`)

| Section | Key | Description |
|---|---|---|
| `broker` | `base_url` | Use the paper-trading URL for testing |
| `signal` | `fast_period` / `slow_period` | MA periods (default 9/21) |
| `signal` | `ma_type` | `"ema"` or `"sma"` |
| `signal` | `bar_timeframe` | `"5Min"`, `"15Min"`, etc. |
| `signal` | `confirmation_bars` | Bars cross must hold before entry |
| `options` | `expiry_dte_min/max` | DTE window for contract selection |
| `options` | `delta_target` | Target delta (default 0.40) |
| `risk` | `max_position_risk_pct` | Max % of portfolio per trade |
| `risk` | `stop_loss_pct` | Premium drop that triggers exit (50% = lose half premium) |
| `risk` | `take_profit_pct` | Premium gain that triggers exit (100% = double) |
| `risk` | `max_daily_loss_pct` | Halt trading after this % portfolio loss in a day |
| `agent` | `dry_run` | `true` = log only, no real orders |

---

## Running

```bash
python -m src.agent
```

---

## Tests

```bash
python -m pytest tests/ -v
```