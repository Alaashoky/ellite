"""Unit tests for OrderManager: modify_stop_loss, validation, bracket orders."""

import math
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_exchange():
    ex = MagicMock()
    ex.get_open_orders.return_value = []
    ex.cancel_order.return_value = True
    ex.place_order.return_value = {
        'id': 'entry-001',
        'symbol': 'BTCUSDT',
        'side': 'buy',
        'status': 'closed',
        'amount': 0.01,
        'price': 50_000.0,
    }
    return ex


@pytest.fixture()
def mock_risk_manager():
    return MagicMock()


@pytest.fixture()
def order_manager(mock_exchange, mock_risk_manager):
    from order_manager import OrderManager
    return OrderManager(exchange=mock_exchange, risk_manager=mock_risk_manager)


# ---------------------------------------------------------------------------
# TestModifyStopLoss
# ---------------------------------------------------------------------------

class TestModifyStopLoss:

    def test_success_path(self, order_manager, mock_exchange):
        """modify_stop_loss retrieves order BEFORE cancel and places new SL."""
        existing_sl = {
            'id': 'sl-001',
            'symbol': 'BTCUSDT',
            'side': 'sell',
            'type': 'stop_loss',
            'amount': 0.05,
            'stop_price': 49_000.0,
        }
        mock_exchange.get_open_orders.return_value = [existing_sl]

        result = order_manager.modify_stop_loss('sl-001', 'BTCUSDT', 48_000.0)

        # open_orders called first so we know what we are cancelling.
        mock_exchange.get_open_orders.assert_called_once_with('BTCUSDT')
        mock_exchange.cancel_order.assert_called_once_with('sl-001', 'BTCUSDT')
        assert result['side'] == 'sell'
        assert result['amount'] == 0.05
        assert result['stop_price'] == 48_000.0

    def test_missing_order_raises_valueerror(self, order_manager, mock_exchange):
        """modify_stop_loss raises ValueError when order is not found."""
        mock_exchange.get_open_orders.return_value = []

        with pytest.raises(ValueError, match="not found"):
            order_manager.modify_stop_loss('nonexistent', 'BTCUSDT', 48_000.0)

        # Cancel must NOT be called when the order was never found.
        mock_exchange.cancel_order.assert_not_called()

    def test_zero_amount_raises_valueerror(self, order_manager, mock_exchange):
        """modify_stop_loss raises ValueError when original amount is zero."""
        existing_sl = {
            'id': 'sl-zero',
            'symbol': 'BTCUSDT',
            'side': 'sell',
            'type': 'stop_loss',
            'amount': 0.0,
            'stop_price': 49_000.0,
        }
        mock_exchange.get_open_orders.return_value = [existing_sl]

        with pytest.raises(ValueError, match="invalid amount"):
            order_manager.modify_stop_loss('sl-zero', 'BTCUSDT', 48_000.0)

    def test_side_preserved_from_original(self, order_manager, mock_exchange):
        """modify_stop_loss must preserve the original order's side (buy SL for short)."""
        existing_sl = {
            'id': 'sl-buy-002',
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'type': 'stop_loss',
            'amount': 0.02,
            'stop_price': 51_000.0,
        }
        mock_exchange.get_open_orders.return_value = [existing_sl]

        result = order_manager.modify_stop_loss('sl-buy-002', 'BTCUSDT', 52_000.0)

        assert result['side'] == 'buy'
        assert result['amount'] == 0.02


# ---------------------------------------------------------------------------
# TestValidateOrder
# ---------------------------------------------------------------------------

class TestValidateOrder:

    def test_rejects_zero_amount(self, order_manager):
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', 0.0)
        assert not ok
        assert 'positive' in reason.lower()

    def test_rejects_negative_amount(self, order_manager):
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', -1.0)
        assert not ok

    def test_rejects_nan_amount(self, order_manager):
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', math.nan)
        assert not ok
        assert 'finite' in reason.lower()

    def test_rejects_inf_amount(self, order_manager):
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', math.inf)
        assert not ok
        assert 'finite' in reason.lower()

    def test_rejects_nan_price(self, order_manager):
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', 0.01, price=math.nan)
        assert not ok
        assert 'finite' in reason.lower()

    def test_rejects_inf_price(self, order_manager):
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', 0.01, price=math.inf)
        assert not ok

    def test_rejects_zero_price(self, order_manager):
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', 0.01, price=0.0)
        assert not ok
        assert 'positive' in reason.lower()

    def test_rejects_negative_price(self, order_manager):
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', 0.01, price=-100.0)
        assert not ok

    def test_rejects_below_min_notional(self, order_manager):
        # BTCUSDT min_notional=10; 0.001 * 1.0 = 0.001 < 10
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', 0.001, price=1.0)
        assert not ok
        assert 'notional' in reason.lower()

    def test_valid_order_passes(self, order_manager):
        # 0.01 BTC @ 50000 = 500 USDT, well above min_notional=10
        ok, reason = order_manager._validate_order('BTCUSDT', 'buy', 0.01, price=50_000.0)
        assert ok
        assert reason == 'OK'

    def test_valid_order_no_price_skips_notional(self, order_manager):
        # Without a price, notional check is skipped.
        ok, _ = order_manager._validate_order('BTCUSDT', 'buy', 0.01)
        assert ok

    def test_rejects_invalid_side(self, order_manager):
        ok, reason = order_manager._validate_order('BTCUSDT', 'long', 0.01)
        assert not ok
        assert 'sell' in reason.lower() or 'side' in reason.lower()

    def test_rejects_unsupported_symbol(self, order_manager):
        ok, reason = order_manager._validate_order('FAKEUSDT', 'buy', 0.01)
        assert not ok
        assert 'symbol' in reason.lower() or 'unsupported' in reason.lower()


# ---------------------------------------------------------------------------
# TestPlaceBracketOrder
# ---------------------------------------------------------------------------

class TestPlaceBracketOrder:

    def test_rejects_empty_tp_list(self, order_manager, mock_exchange):
        """No TP targets must raise before placing entry."""
        with pytest.raises(ValueError, match="valid take-profit"):
            order_manager.place_bracket_order(
                'BTCUSDT', 'buy', 0.01, 50_000.0, 49_000.0, []
            )
        mock_exchange.place_order.assert_not_called()

    def test_rejects_all_invalid_tp_values(self, order_manager, mock_exchange):
        """NaN / inf / negative TPs are all stripped; result is no valid TPs."""
        with pytest.raises(ValueError, match="valid take-profit"):
            order_manager.place_bracket_order(
                'BTCUSDT', 'buy', 0.01, 50_000.0, 49_000.0,
                [math.nan, math.inf, -1.0, 0.0]
            )
        mock_exchange.place_order.assert_not_called()

    def test_rejects_tp_producing_under_min_amount(self, order_manager, mock_exchange):
        """TP split that falls below min_order_size must raise before entry."""
        # 0.001 / 3 ≈ 0.000333 < min_order_size 0.001
        with pytest.raises(ValueError, match="min_order_size"):
            order_manager.place_bracket_order(
                'BTCUSDT', 'buy', 0.001, 50_000.0, 49_000.0,
                [51_000.0, 52_000.0, 53_000.0]
            )
        mock_exchange.place_order.assert_not_called()

    def test_sl_tp_side_correct_for_buy(self, order_manager, mock_exchange):
        """For a buy entry, SL and all TP orders must use side='sell'."""
        result = order_manager.place_bracket_order(
            'BTCUSDT', 'buy', 0.01, 50_000.0, 49_000.0,
            [51_000.0, 52_000.0]
        )
        assert result['stop_loss']['side'] == 'sell'
        for tp in result['take_profits']:
            assert tp['side'] == 'sell'

    def test_sl_tp_side_correct_for_sell(self, order_manager, mock_exchange):
        """For a sell entry, SL and all TP orders must use side='buy'."""
        mock_exchange.place_order.return_value = {
            'id': 'entry-short-001',
            'symbol': 'BTCUSDT',
            'side': 'sell',
            'status': 'closed',
            'amount': 0.01,
            'price': 50_000.0,
        }
        result = order_manager.place_bracket_order(
            'BTCUSDT', 'sell', 0.01, 50_000.0, 51_000.0,
            [49_000.0, 48_000.0]
        )
        assert result['stop_loss']['side'] == 'buy'
        for tp in result['take_profits']:
            assert tp['side'] == 'buy'

    def test_deduplicates_tp_targets(self, order_manager, mock_exchange):
        """Duplicate TP prices should produce only one TP order each."""
        result = order_manager.place_bracket_order(
            'BTCUSDT', 'buy', 0.01, 50_000.0, 49_000.0,
            [51_000.0, 51_000.0, 52_000.0]  # 51k duplicated
        )
        assert len(result['take_profits']) == 2

    def test_success_structure(self, order_manager, mock_exchange):
        """Successful bracket returns entry, stop_loss, and take_profits keys."""
        result = order_manager.place_bracket_order(
            'BTCUSDT', 'buy', 0.01, 50_000.0, 49_000.0,
            [51_000.0]
        )
        assert 'entry' in result
        assert 'stop_loss' in result
        assert 'take_profits' in result
        assert len(result['take_profits']) == 1


# ---------------------------------------------------------------------------
# TestExchangeConnectorPaperCaching
# ---------------------------------------------------------------------------

class TestExchangeConnectorPaperCaching:

    def test_paper_public_exchange_created_once(self):
        """_get_paper_public_exchange returns the same instance every call."""
        from exchange_connector import ExchangeConnector
        ex = ExchangeConnector(paper_mode=True)

        mock_ccxt_instance = MagicMock()
        with patch('ccxt.binance', return_value=mock_ccxt_instance) as mock_binance:
            inst1 = ex._get_paper_public_exchange()
            inst2 = ex._get_paper_public_exchange()

        # Constructor should be called exactly once.
        assert mock_binance.call_count == 1
        assert inst1 is inst2

    def test_get_ticker_reuses_cached_exchange(self):
        """get_ticker in paper mode uses cached exchange, not a new one each call."""
        from exchange_connector import ExchangeConnector
        ex = ExchangeConnector(paper_mode=True)

        mock_public = MagicMock()
        mock_public.fetch_ticker.return_value = {'last': 50_000.0}

        with patch.object(ex, '_get_paper_public_exchange', return_value=mock_public) as mock_getter:
            ex.get_ticker('BTCUSDT')
            ex.get_ticker('BTCUSDT')

        # The accessor is called each time get_ticker is invoked…
        assert mock_getter.call_count == 2
        # …but the underlying fetch is made on the same (cached) object both times.
        assert mock_public.fetch_ticker.call_count == 2

    def test_get_ohlcv_reuses_cached_exchange(self):
        """get_ohlcv in paper mode uses cached exchange, not a new one each call."""
        from exchange_connector import ExchangeConnector
        ex = ExchangeConnector(paper_mode=True)

        mock_public = MagicMock()
        mock_public.fetch_ohlcv.return_value = []

        with patch.object(ex, '_get_paper_public_exchange', return_value=mock_public):
            ex.get_ohlcv('BTCUSDT', limit=10)
            ex.get_ohlcv('BTCUSDT', limit=10)

        assert mock_public.fetch_ohlcv.call_count == 2

    def test_place_order_rejects_zero_amount(self):
        """place_order raises ValueError for zero amount."""
        from exchange_connector import ExchangeConnector
        ex = ExchangeConnector(paper_mode=True)
        with pytest.raises(ValueError, match="Invalid order amount"):
            ex.place_order('BTCUSDT', 'buy', 0.0)

    def test_place_order_rejects_negative_amount(self):
        """place_order raises ValueError for negative amount."""
        from exchange_connector import ExchangeConnector
        ex = ExchangeConnector(paper_mode=True)
        with pytest.raises(ValueError, match="Invalid order amount"):
            ex.place_order('BTCUSDT', 'buy', -1.0)

    def test_place_order_rejects_nan_amount(self):
        """place_order raises ValueError for NaN amount."""
        from exchange_connector import ExchangeConnector
        ex = ExchangeConnector(paper_mode=True)
        with pytest.raises(ValueError, match="Invalid order amount"):
            ex.place_order('BTCUSDT', 'buy', math.nan)

    def test_place_order_rejects_invalid_price(self):
        """place_order raises ValueError for NaN price."""
        from exchange_connector import ExchangeConnector
        ex = ExchangeConnector(paper_mode=True)
        with pytest.raises(ValueError, match="Invalid order price"):
            ex.place_order('BTCUSDT', 'buy', 0.01, price=math.nan)
