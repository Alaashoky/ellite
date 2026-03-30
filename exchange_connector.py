"""Exchange connector supporting both live ccxt and paper trading modes."""

import time
import logging
import uuid
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExchangeConnector:
    """
    Unified interface for live (Binance via ccxt) and paper trading.

    Paper mode requires no API keys and simulates fills with 0.05% slippage.
    """

    SUPPORTED_SYMBOLS: Dict[str, str] = {
        'BTCUSDT': 'BTC/USDT',
        'XAUUSDT': 'XAU/USDT',
    }
    SLIPPAGE = 0.0005  # 0.05%

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper_mode: bool = True,
        initial_balance: float = 10_000.0,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_mode = paper_mode
        self._exchange = None

        if paper_mode:
            self._paper_balance: Dict[str, float] = {
                'USDT': initial_balance, 'BTC': 0.0, 'XAU': 0.0
            }
            self._paper_orders: List[Dict] = []
            self._paper_public_exchange = None  # lazily cached public client
        else:
            self._init_live_exchange()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_live_exchange(self) -> None:
        try:
            import ccxt
            self._exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            })
        except ImportError as e:
            raise ImportError("ccxt is required for live trading.") from e

    def _get_paper_public_exchange(self):
        """Return a cached public ccxt.binance instance for paper-mode data fetches."""
        if self._paper_public_exchange is None:
            import ccxt
            self._paper_public_exchange = ccxt.binance({'enableRateLimit': True})
        return self._paper_public_exchange

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '15m',
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ccxt_symbol = self.SUPPORTED_SYMBOLS.get(symbol, symbol.replace('USDT', '/USDT'))

        if self.paper_mode:
            # Reuse cached public exchange instance (no auth needed for public endpoints).
            try:
                ex = self._get_paper_public_exchange()
                raw = self._retry_with_backoff(
                    ex.fetch_ohlcv, ccxt_symbol, timeframe, limit=limit
                )
            except Exception as e:
                logger.warning(f"get_ohlcv paper fallback failed: {e}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        else:
            raw = self._retry_with_backoff(
                self._exchange.fetch_ohlcv, ccxt_symbol, timeframe, limit=limit
            )

        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df

    def get_ticker(self, symbol: str) -> Dict:
        """Return latest ticker data for a symbol."""
        ccxt_symbol = self.SUPPORTED_SYMBOLS.get(symbol, symbol)
        if self.paper_mode:
            try:
                ex = self._get_paper_public_exchange()
                return self._retry_with_backoff(ex.fetch_ticker, ccxt_symbol)
            except Exception as e:
                logger.warning(f"get_ticker failed: {e}")
                return {}
        return self._retry_with_backoff(self._exchange.fetch_ticker, ccxt_symbol)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = 'market',
    ) -> Dict:
        """
        Place an order.

        Performs defensive validation (finite/positive amount and price) before
        forwarding to paper simulation or the live exchange.
        Paper mode: fills immediately with simulated slippage.
        """
        import math
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError(f"Invalid order amount: {amount}")
        if price is not None and (not math.isfinite(price) or price <= 0):
            raise ValueError(f"Invalid order price: {price}")
        if self.paper_mode:
            return self._paper_place_order(symbol, side, amount, price, order_type)
        ccxt_symbol = self.SUPPORTED_SYMBOLS.get(symbol, symbol)
        if order_type == 'market':
            return self._retry_with_backoff(
                self._exchange.create_market_order, ccxt_symbol, side, amount
            )
        return self._retry_with_backoff(
            self._exchange.create_limit_order, ccxt_symbol, side, amount, price
        )

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True on success."""
        if self.paper_mode:
            before = len(self._paper_orders)
            self._paper_orders = [
                o for o in self._paper_orders
                if not (o['id'] == order_id and o['symbol'] == symbol)
            ]
            return len(self._paper_orders) < before
        ccxt_symbol = self.SUPPORTED_SYMBOLS.get(symbol, symbol)
        try:
            self._retry_with_backoff(self._exchange.cancel_order, order_id, ccxt_symbol)
            return True
        except Exception as e:
            logger.error(f"cancel_order failed: {e}")
            return False

    def get_balance(self) -> Dict:
        """Return account balance dict."""
        if self.paper_mode:
            return {'free': dict(self._paper_balance), 'total': dict(self._paper_balance)}
        try:
            return self._retry_with_backoff(self._exchange.fetch_balance)
        except Exception as e:
            logger.error(f"get_balance failed: {e}")
            return {}

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Return list of open orders (optionally filtered by symbol)."""
        if self.paper_mode:
            if symbol:
                return [o for o in self._paper_orders if o['symbol'] == symbol]
            return list(self._paper_orders)
        ccxt_symbol = self.SUPPORTED_SYMBOLS.get(symbol, symbol) if symbol else None
        try:
            return self._retry_with_backoff(self._exchange.fetch_open_orders, ccxt_symbol)
        except Exception as e:
            logger.error(f"get_open_orders failed: {e}")
            return []

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Return order details by ID."""
        if self.paper_mode:
            for o in self._paper_orders:
                if o['id'] == order_id and o['symbol'] == symbol:
                    return o
            return {'id': order_id, 'symbol': symbol, 'status': 'closed'}
        ccxt_symbol = self.SUPPORTED_SYMBOLS.get(symbol, symbol)
        return self._retry_with_backoff(self._exchange.fetch_order, order_id, ccxt_symbol)

    # ------------------------------------------------------------------
    # Paper trading internals
    # ------------------------------------------------------------------

    def _paper_place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float],
        order_type: str,
    ) -> Dict:
        """Simulate an order fill with slippage."""
        try:
            ticker = self.get_ticker(symbol)
            market_price = float(ticker.get('last', price or 0))
        except Exception:
            market_price = price or 0.0

        slippage = self.SLIPPAGE if side == 'buy' else -self.SLIPPAGE
        fill_price = market_price * (1 + slippage) if order_type == 'market' else (price or market_price)

        cost = fill_price * amount
        asset = symbol.replace('USDT', '')

        if side == 'buy':
            if self._paper_balance.get('USDT', 0) < cost:
                raise ValueError(f"Insufficient USDT balance. Need {cost:.2f}, have {self._paper_balance.get('USDT', 0):.2f}")
            self._paper_balance['USDT'] = self._paper_balance.get('USDT', 0) - cost
            self._paper_balance[asset] = self._paper_balance.get(asset, 0) + amount
        else:  # sell
            if self._paper_balance.get(asset, 0) < amount:
                raise ValueError(f"Insufficient {asset} balance.")
            self._paper_balance[asset] = self._paper_balance.get(asset, 0) - amount
            self._paper_balance['USDT'] = self._paper_balance.get('USDT', 0) + cost

        order = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'amount': amount,
            'price': fill_price,
            'cost': cost,
            'status': 'closed',
            'filled': amount,
            'remaining': 0.0,
            'timestamp': int(time.time() * 1000),
        }
        logger.debug(f"[PAPER] {side.upper()} {amount} {symbol} @ {fill_price:.4f}")
        return order

    # ------------------------------------------------------------------
    # Retry utility
    # ------------------------------------------------------------------

    def _retry_with_backoff(self, func, *args, max_retries: int = 3, **kwargs):
        """Call `func` with exponential backoff on transient errors."""
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"All {max_retries} retries failed: {last_exc}") from last_exc
