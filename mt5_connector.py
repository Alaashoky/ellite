"""
mt5_connector.py
Handles all MetaTrader 5 interactions:
  - Connection / login
  - Historical OHLCV data fetching (from 2020-01-01 to any date)
  - Live order placement (market buy/sell, SL, TP)
  - Position management

Note: The MetaTrader5 Python API only works on Windows.
On Linux/macOS this module raises an ImportError with a helpful message.
"""

import logging
import platform
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform guard — MT5 Python API is Windows-only
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5  # type: ignore
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None  # type: ignore

MT5_TIMEFRAME_MAP: Dict[str, Any] = {}
if _MT5_AVAILABLE:
    MT5_TIMEFRAME_MAP = {
        "1m":  mt5.TIMEFRAME_M1,
        "5m":  mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h":  mt5.TIMEFRAME_H1,
        "4h":  mt5.TIMEFRAME_H4,
        "1d":  mt5.TIMEFRAME_D1,
        "1w":  mt5.TIMEFRAME_W1,
    }


def _require_mt5() -> None:
    """Raise a helpful ImportError when MetaTrader5 is not available."""
    if not _MT5_AVAILABLE:
        raise ImportError(
            "The MetaTrader5 Python package is not installed or not supported on "
            f"this platform ({platform.system()}). "
            "The MT5 API only works on Windows. "
            "Install it with:  pip install MetaTrader5>=5.0.45\n"
            "Then ensure that MetaTrader 5 terminal is installed and running."
        )


class MT5Connector:
    """
    Full MetaTrader 5 connector.

    Handles connection/login, historical OHLCV data fetching with chunked
    pagination (year-by-year) to reliably retrieve data from 2020 onward,
    live/paper order placement, and position management.
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str = "",
    ) -> None:
        _require_mt5()
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise the MT5 terminal and log in."""
        _require_mt5()
        kwargs: Dict[str, Any] = {}
        if self.path:
            kwargs["path"] = self.path

        if not mt5.initialize(**kwargs):
            error = mt5.last_error()
            logger.error(f"MT5 initialize() failed: {error}")
            return False

        authorized = mt5.login(
            login=self.login,
            password=self.password,
            server=self.server,
        )
        if not authorized:
            error = mt5.last_error()
            logger.error(f"MT5 login() failed for account {self.login}: {error}")
            mt5.shutdown()
            return False

        info = mt5.account_info()
        logger.info(
            f"MT5 connected — account={self.login}, "
            f"server={self.server}, "
            f"balance={getattr(info, 'balance', 'N/A')}"
        )
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 terminal connection."""
        if _MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected.")

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "MT5Connector is not connected. Call connect() first."
            )

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns: time, open, high, low, close, volume.

        Fetches data in year-long chunks to reliably retrieve all history
        from start_date (e.g. "2020-01-01") to end_date (defaults to today).

        All timestamps are normalised to UTC.
        """
        _require_mt5()
        self._ensure_connected()

        tf = MT5_TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. "
                f"Supported: {list(MT5_TIMEFRAME_MAP.keys())}"
            )

        # Ensure symbol is available in the Market Watch
        if not mt5.symbol_select(symbol, True):
            raise ValueError(
                f"Symbol '{symbol}' not found in MT5 terminal. "
                "Check the symbol name (e.g. 'EURUSD', 'XAUUSD')."
            )

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            raise ValueError(
                f"start_date '{start_date}' is not a valid date. "
                "Expected format: YYYY-MM-DD (e.g. '2020-01-01')."
            )

        try:
            end_dt = (
                datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if end_date
                else datetime.now(timezone.utc)
            )
        except ValueError:
            raise ValueError(
                f"end_date '{end_date}' is not a valid date. "
                "Expected format: YYYY-MM-DD (e.g. '2023-12-31')."
            )

        all_frames: List[pd.DataFrame] = []
        chunk_start = start_dt

        logger.info(
            f"Fetching {symbol} {timeframe} from {start_date} to "
            f"{end_date or 'today'} in annual chunks…"
        )

        # Chunk year-by-year to avoid broker candle limits
        while chunk_start < end_dt:
            chunk_end = min(
                chunk_start.replace(year=chunk_start.year + 1),
                end_dt,
            )

            rates = None
            for attempt in range(3):
                rates = mt5.copy_rates_range(symbol, tf, chunk_start, chunk_end)
                if rates is not None:
                    break
                err = mt5.last_error()
                logger.warning(
                    f"copy_rates_range attempt {attempt + 1} failed: {err}. "
                    "Retrying in 5s…"
                )
                time.sleep(5)

            if rates is None or len(rates) == 0:
                logger.warning(
                    f"No data for {symbol} {chunk_start.date()} – "
                    f"{chunk_end.date()}, skipping chunk."
                )
            else:
                df_chunk = pd.DataFrame(rates)
                all_frames.append(df_chunk)
                logger.debug(
                    f"  chunk {chunk_start.date()} – {chunk_end.date()}: "
                    f"{len(df_chunk)} rows"
                )

            chunk_start = chunk_end

        if not all_frames:
            raise ValueError(
                f"No OHLCV data returned for {symbol} between "
                f"{start_date} and {end_date or 'today'}."
            )

        df = pd.concat(all_frames, ignore_index=True)

        # MT5 returns 'time' as POSIX timestamps (seconds, UTC)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # Keep only the standard OHLCV columns plus time
        cols = ["time", "open", "high", "low", "close", "tick_volume"]
        df = df[[c for c in cols if c in df.columns]].copy()
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        df.drop_duplicates(subset=["time"], keep="first", inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}.")
        return df

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "ellite_bot",
    ) -> Dict[str, Any]:
        """
        Place a market buy or sell order.

        Args:
            symbol:     Instrument name, e.g. "EURUSD".
            order_type: "buy" or "sell".
            volume:     Lot size.
            price:      For market orders this is ignored (current ask/bid used).
            sl:         Stop-loss price (optional).
            tp:         Take-profit price (optional).
            comment:    Order comment visible in terminal.

        Returns:
            Dict with order result details.
        """
        _require_mt5()
        self._ensure_connected()

        order_type_lower = order_type.lower()
        if order_type_lower not in ("buy", "sell"):
            raise ValueError(f"order_type must be 'buy' or 'sell', got '{order_type}'.")

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Could not get tick data for '{symbol}'.")

        if order_type_lower == "buy":
            mt5_order_type = mt5.ORDER_TYPE_BUY
            exec_price = tick.ask
        else:
            mt5_order_type = mt5.ORDER_TYPE_SELL
            exec_price = tick.bid

        request: Dict[str, Any] = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   symbol,
            "volume":   float(volume),
            "type":     mt5_order_type,
            "price":    exec_price,
            "deviation": 20,
            "magic":    234000,
            "comment":  comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)

        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            raise RuntimeError(f"order_send() returned None — MT5 error: {error}")

        result_dict = result._asdict()
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"Order failed for {symbol} {order_type} {volume} lots — "
                f"retcode={result.retcode}, comment={result.comment}"
            )
        else:
            logger.info(
                f"Order placed: {symbol} {order_type} {volume} lots "
                f"@ {result.price} ticket={result.order}"
            )
        return dict(result_dict)

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_open_positions(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return a list of open positions, optionally filtered by symbol."""
        _require_mt5()
        self._ensure_connected()

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []
        return [p._asdict() for p in positions]

    def close_position(self, ticket: int) -> bool:
        """
        Close an open position identified by its ticket number.

        Returns True if the close order was accepted, False otherwise.
        """
        _require_mt5()
        self._ensure_connected()

        positions = mt5.positions_get()
        if not positions:
            logger.warning(f"No open positions found when trying to close ticket={ticket}.")
            return False

        pos = next((p for p in positions if p.ticket == ticket), None)
        if pos is None:
            logger.warning(f"Position ticket={ticket} not found.")
            return False

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            logger.error(f"Could not get tick for {pos.symbol}.")
            return False

        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            close_price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            close_price = tick.ask

        request: Dict[str, Any] = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    pos.symbol,
            "volume":    pos.volume,
            "type":      close_type,
            "position":  ticket,
            "price":     close_price,
            "deviation": 20,
            "magic":     234000,
            "comment":   "ellite_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error() if result is None else result.comment
            logger.error(f"Failed to close position {ticket}: {error}")
            return False

        logger.info(f"Position {ticket} ({pos.symbol}) closed successfully.")
        return True
