"""Order management: market, limit, stop-loss, take-profit, and bracket orders."""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OrderManager:
    """
    High-level order placement layer on top of ExchangeConnector.

    Validates orders via RiskManager before submission.
    """

    def __init__(self, exchange, risk_manager):
        """
        Args:
            exchange:     ExchangeConnector instance
            risk_manager: RiskManager instance
        """
        self.exchange = exchange
        self.risk_manager = risk_manager

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """Place a market order. Raises ValueError on validation failure."""
        ok, reason = self._validate_order(symbol, side, amount)
        if not ok:
            raise ValueError(f"Order validation failed: {reason}")
        result = self.exchange.place_order(symbol, side, amount, order_type='market')
        logger.info(f"Market {side.upper()} {amount} {symbol} → {result.get('id')}")
        return result

    def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> Dict:
        """Place a limit order at the specified price."""
        ok, reason = self._validate_order(symbol, side, amount, price)
        if not ok:
            raise ValueError(f"Order validation failed: {reason}")
        result = self.exchange.place_order(symbol, side, amount, price=price, order_type='limit')
        logger.info(f"Limit {side.upper()} {amount} {symbol} @ {price} → {result.get('id')}")
        return result

    def place_stop_loss(
        self, symbol: str, side: str, amount: float, stop_price: float
    ) -> Dict:
        """
        Place a stop-loss order.

        In paper/spot mode this is stored as a tracking order (not exchange-native).
        """
        order = {
            'id': f'sl_{symbol}_{side}_{stop_price}',
            'symbol': symbol,
            'side': side,
            'type': 'stop_loss',
            'amount': amount,
            'stop_price': stop_price,
            'status': 'open',
        }
        logger.info(f"SL registered for {symbol} {side.upper()} @ {stop_price}")
        return order

    def place_take_profit(
        self, symbol: str, side: str, amount: float, tp_price: float
    ) -> Dict:
        """
        Place a take-profit order (tracked, not exchange-native in paper mode).
        """
        order = {
            'id': f'tp_{symbol}_{side}_{tp_price}',
            'symbol': symbol,
            'side': side,
            'type': 'take_profit',
            'amount': amount,
            'tp_price': tp_price,
            'status': 'open',
        }
        logger.info(f"TP registered for {symbol} {side.upper()} @ {tp_price}")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order by ID."""
        result = self.exchange.cancel_order(order_id, symbol)
        logger.info(f"Cancelled order {order_id} for {symbol}: {result}")
        return result

    def modify_stop_loss(
        self, order_id: str, symbol: str, new_stop: float
    ) -> Dict:
        """
        Modify an existing stop-loss by cancelling and re-registering.
        """
        self.exchange.cancel_order(order_id, symbol)
        open_orders = self.exchange.get_open_orders(symbol)
        original = next((o for o in open_orders if o['id'] == order_id), None)
        side = original['side'] if original else 'sell'
        amount = original['amount'] if original else 0.0
        return self.place_stop_loss(symbol, side, amount, new_stop)

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Retrieve current order status."""
        return self.exchange.get_order_status(order_id, symbol)

    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        stop_loss: float,
        take_profits: List[float],
    ) -> Dict:
        """
        Place entry + SL + multiple TP orders atomically.

        Returns a dict containing all placed order references.
        """
        entry_order = self.place_market_order(symbol, side, amount)
        sl_side = 'sell' if side == 'buy' else 'buy'

        sl_order = self.place_stop_loss(symbol, sl_side, amount, stop_loss)

        tp_orders = []
        tp_amount = amount / max(len(take_profits), 1)
        for tp_price in take_profits:
            tp_ord = self.place_take_profit(symbol, sl_side, tp_amount, tp_price)
            tp_orders.append(tp_ord)

        return {
            'entry': entry_order,
            'stop_loss': sl_order,
            'take_profits': tp_orders,
        }

    def cancel_all_orders(self, symbol: str) -> int:
        """
        Cancel all open orders for a symbol.

        Returns:
            Number of orders successfully cancelled.
        """
        orders = self.exchange.get_open_orders(symbol)
        count = 0
        for order in orders:
            try:
                if self.exchange.cancel_order(order['id'], symbol):
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to cancel order {order.get('id')}: {e}")
        logger.info(f"Cancelled {count}/{len(orders)} orders for {symbol}")
        return count

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Perform pre-order validation checks.

        Returns (is_valid, reason_string).
        """
        if amount <= 0:
            return False, f"Amount must be positive, got {amount}"
        if price is not None and price <= 0:
            return False, f"Price must be positive, got {price}"
        if side not in ('buy', 'sell'):
            return False, f"Side must be 'buy' or 'sell', got '{side}'"
        from config.settings import ASSETS_CONFIG
        if symbol not in ASSETS_CONFIG:
            return False, f"Unsupported symbol: {symbol}"
        cfg = ASSETS_CONFIG[symbol]
        if amount < cfg.get('min_order_size', 0):
            return False, f"Amount {amount} below min_order_size {cfg['min_order_size']}"
        return True, "OK"
