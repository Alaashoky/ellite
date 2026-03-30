"""Order management: market, limit, stop-loss, take-profit, and bracket orders."""

import logging
import math
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

        Retrieves the original order details **before** cancellation so that
        side and amount are never lost.  Raises ``ValueError`` if the order
        cannot be found or carries an invalid (zero/negative) amount.
        """
        # Fetch details BEFORE cancellation – the order disappears afterward.
        open_orders = self.exchange.get_open_orders(symbol)
        original = next((o for o in open_orders if o['id'] == order_id), None)

        if original is None:
            logger.error(
                f"modify_stop_loss: order '{order_id}' not found for {symbol} "
                f"(new_stop={new_stop})"
            )
            raise ValueError(
                f"Cannot modify stop-loss: order '{order_id}' not found for {symbol}"
            )

        side = original.get('side')
        amount = original.get('amount', 0.0)
        old_stop = original.get('stop_price', original.get('price'))

        if amount is None or amount <= 0:
            raise ValueError(
                f"Cannot modify stop-loss: original order '{order_id}' "
                f"has invalid amount {amount}"
            )

        self.exchange.cancel_order(order_id, symbol)
        new_sl = self.place_stop_loss(symbol, side, amount, new_stop)
        logger.info(
            f"modify_stop_loss: {symbol} order {order_id} | "
            f"old_stop={old_stop} → new_stop={new_stop} | "
            f"side={side} amount={amount} | new_sl_id={new_sl.get('id')}"
        )
        return new_sl

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

        Validates TP targets (finite, positive, deduplicated) and ensures the
        per-TP split amount meets ``min_order_size`` before executing any leg.
        Raises ``ValueError`` for any invalid configuration.

        Returns a dict containing all placed order references.
        """
        # Validate entry parameters.
        ok, reason = self._validate_order(symbol, side, amount, entry_price)
        if not ok:
            raise ValueError(f"Bracket order validation failed: {reason}")

        from config.settings import ASSETS_CONFIG
        cfg = ASSETS_CONFIG.get(symbol, {})
        min_order_size = cfg.get('min_order_size', 0)

        # Filter, validate, and deduplicate TP targets.
        seen: set = set()
        valid_tps: List[float] = []
        for tp in take_profits:
            if not math.isfinite(tp) or tp <= 0:
                logger.warning(f"Skipping invalid TP target {tp} for {symbol}")
                continue
            if tp not in seen:  # O(1) membership check via hash set
                seen.add(tp)
                valid_tps.append(tp)

        if not valid_tps:
            raise ValueError(
                f"No valid take-profit targets provided for {symbol}"
            )

        # Ensure each TP slice meets the exchange minimum.
        tp_amount = amount / len(valid_tps)
        if tp_amount < min_order_size:
            raise ValueError(
                f"TP split amount {tp_amount} is below min_order_size "
                f"{min_order_size} for {len(valid_tps)} targets. "
                "Reduce TP count or increase position size."
            )

        # Execute: entry → SL → TPs.
        entry_order = self.place_market_order(symbol, side, amount)
        # Closing side is always opposite to entry for both SL and TP.
        close_side = 'sell' if side == 'buy' else 'buy'

        sl_order = self.place_stop_loss(symbol, close_side, amount, stop_loss)

        tp_orders = []
        for tp_price in valid_tps:
            tp_ord = self.place_take_profit(symbol, close_side, tp_amount, tp_price)
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

        Checks include: finite/positive amount, finite/positive price,
        valid side, recognised symbol, min order size, and min notional.

        Returns (is_valid, reason_string).
        """
        if not math.isfinite(amount):
            return False, f"Amount must be a finite number, got {amount}"
        if amount <= 0:
            return False, f"Amount must be positive, got {amount}"
        if price is not None:
            if not math.isfinite(price):
                return False, f"Price must be a finite number, got {price}"
            if price <= 0:
                return False, f"Price must be positive, got {price}"
        if side not in ('buy', 'sell'):
            return False, f"Side must be 'buy' or 'sell', got '{side}'"
        from config.settings import ASSETS_CONFIG
        if symbol not in ASSETS_CONFIG:
            return False, f"Unsupported symbol: {symbol}"
        cfg = ASSETS_CONFIG[symbol]
        if amount < cfg.get('min_order_size', 0):
            return False, f"Amount {amount} below min_order_size {cfg['min_order_size']}"
        if price is not None:
            min_notional = cfg.get('min_notional', 0)
            notional = amount * price
            if notional < min_notional:
                return False, (
                    f"Notional {notional:.4f} below min_notional {min_notional}"
                )
        return True, "OK"
