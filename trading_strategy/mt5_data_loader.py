"""
trading_strategy/mt5_data_loader.py

Adapter so TradingStrategy / ModelTrainer can use MT5 as their data source.
Drop-in replacement for the existing DataLoader, backed by MetaTrader 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from mt5_connector import MT5Connector


class MT5DataLoader:
    """
    Drop-in replacement for the existing DataLoader, backed by MT5.

    Normalises the raw MT5 DataFrame so that downstream code
    (TradingStrategy, ModelTrainer, etc.) receives data in the same
    format as the ccxt-based DataLoader:

        Index : DatetimeIndex named 'timestamp' (UTC)
        Cols  : open, high, low, close, volume
    """

    DEFAULT_START_DATE = "2020-01-01"

    def __init__(self, connector: "MT5Connector") -> None:
        self.connector = connector

    def load_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from MT5 and return it in the standard format.

        Args:
            symbol:     MT5 instrument name, e.g. "EURUSD", "XAUUSD".
            timeframe:  Timeframe string matching MT5_TIMEFRAME_MAP keys,
                        e.g. "1m", "5m", "15m", "1h", "4h", "1d".
            start_date: ISO date string "YYYY-MM-DD".
                        Defaults to "2020-01-01".
            end_date:   ISO date string "YYYY-MM-DD".
                        Defaults to today (MT5Connector handles this).

        Returns:
            DataFrame indexed by 'timestamp' (UTC DatetimeIndex) with
            columns: open, high, low, close, volume.
        """
        start = start_date or self.DEFAULT_START_DATE
        df = self.connector.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start,
            end_date=end_date,
        )

        # Normalise column names to what the rest of the codebase expects
        df.rename(columns={"time": "timestamp"}, inplace=True)
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"

        # Ensure expected columns exist
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(
                    f"MT5DataLoader: expected column '{col}' not found in "
                    f"data returned by MT5Connector for {symbol}."
                )

        return df[["open", "high", "low", "close", "volume"]]
