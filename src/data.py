"""5-minute OHLCV data fetcher using the Alpaca Markets API."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Optional import — allow running without alpaca-py for testing/dry-run
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    _ALPACA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ALPACA_AVAILABLE = False


def _timeframe_from_string(tf_str: str):
    """Convert a timeframe string like '5Min' to an Alpaca TimeFrame object."""
    if not _ALPACA_AVAILABLE:
        raise ImportError("alpaca-py is required to fetch live data")
    mapping = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    if tf_str not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf_str}. Choose from {list(mapping)}")
    return mapping[tf_str]


class DataFetcher:
    """Fetches historical OHLCV bars from Alpaca."""

    def __init__(self, api_key: str, api_secret: str):
        if not _ALPACA_AVAILABLE:
            raise ImportError("alpaca-py is required. Install with: pip install alpaca-py")
        self._client = StockHistoricalDataClient(api_key, api_secret)

    def get_bars(
        self,
        symbol: str,
        timeframe_str: str,
        lookback_bars: int,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of OHLCV bars for *symbol*.

        Columns: open, high, low, close, volume
        Index: UTC-aware datetime
        """
        tf = _timeframe_from_string(timeframe_str)
        if end is None:
            end = datetime.now(tz=timezone.utc)

        # Request extra bars to account for non-trading gaps
        buffer_factor = 3
        minutes_per_bar = _bar_minutes(timeframe_str)
        start = end - timedelta(minutes=lookback_bars * minutes_per_bar * buffer_factor)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
        )

        bars = self._client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            logger.warning("No bars returned for %s", symbol)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Flatten multi-index if present
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.sort_index()
        df = df.tail(lookback_bars)
        logger.debug("Fetched %d bars for %s (%s)", len(df), symbol, timeframe_str)
        return df


def _bar_minutes(timeframe_str: str) -> int:
    """Return the number of minutes represented by a timeframe string."""
    mapping = {
        "1Min": 1,
        "5Min": 5,
        "15Min": 15,
        "30Min": 30,
        "1Hour": 60,
        "1Day": 390,
    }
    return mapping.get(timeframe_str, 5)
