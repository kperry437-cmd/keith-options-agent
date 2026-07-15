"""
data.py – Fetch historical 5-minute OHLCV bars from Alpaca.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")
_UTC = pytz.UTC


def get_bars(symbol: str, n_bars: int, api_key: str, secret_key: str) -> pd.DataFrame:
    """Return the last *n_bars* 5-minute bars for *symbol* as a DataFrame.

    Columns: open, high, low, close, volume (index: timestamp UTC).

    Requests a lookback window large enough to obtain *n_bars* non-holiday
    trading bars even if the market has been closed over a weekend.

    Raises:
        ValueError: if the returned data has fewer bars than *n_bars*.
    """
    # Import here so the module can be imported in tests without alpaca-py installed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(api_key, secret_key)

    end = datetime.now(_UTC)
    # 5-min bars: ~78/day; fetch 4× more calendar days to cover weekends/holidays
    lookback_days = max(10, (n_bars // 78 + 1) * 4)
    start = end - timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        adjustment="split",
    )

    logger.debug("Fetching bars for %s (last %d bars)…", symbol, n_bars)
    bars = client.get_stock_bars(request)
    df: pd.DataFrame = bars.df

    if df.empty:
        raise ValueError(f"No bars returned for {symbol}")

    # Multi-index → single-index when multiple symbols were returned
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)

    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index().tail(n_bars)

    if len(df) < n_bars:
        logger.warning(
            "%s: only %d bars available (requested %d)", symbol, len(df), n_bars
        )

    return df[["open", "high", "low", "close", "volume"]]
