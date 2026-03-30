"""Unit tests for ExchangeConnector paper trading mode."""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def paper_exchange():
    """ExchangeConnector in paper mode with 10,000 USDT starting balance."""
    from exchange_connector import ExchangeConnector
    return ExchangeConnector(paper_mode=True, initial_balance=10_000.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExchangeConnectorPaper:

    def test_initial_balance(self, paper_exchange):
        balance = paper_exchange.get_balance()
        assert balance['free']['USDT'] == pytest.approx(10_000.0)
        assert balance['free']['BTC'] == pytest.approx(0.0)

    def test_supported_symbols(self, paper_exchange):
        assert 'BTCUSDT' in paper_exchange.SUPPORTED_SYMBOLS
        assert 'XAUUSDT' in paper_exchange.SUPPORTED_SYMBOLS

    def test_paper_buy_order(self, paper_exchange, monkeypatch):
        """Buy order should deduct USDT and credit BTC."""
        # Patch get_ticker to avoid network call
        monkeypatch.setattr(
            paper_exchange, 'get_ticker',
            lambda sym: {'last': 50_000.0}
        )
        order = paper_exchange.place_order('BTCUSDT', 'buy', 0.1)
        assert order['side'] == 'buy'
        assert order['amount'] == 0.1
        assert order['status'] == 'closed'
        assert order['filled'] == 0.1
        balance = paper_exchange.get_balance()
        assert balance['free']['BTC'] == pytest.approx(0.1, abs=1e-8)
        assert balance['free']['USDT'] < 10_000.0

    def test_paper_sell_order(self, paper_exchange, monkeypatch):
        """Sell order should deduct BTC and credit USDT."""
        monkeypatch.setattr(paper_exchange, 'get_ticker', lambda sym: {'last': 50_000.0})
        # Use 0.19 BTC so buy cost (0.19 * 50000 * 1.0005 ≈ 9504.75) stays within 10k balance.
        paper_exchange.place_order('BTCUSDT', 'buy', 0.19)
        bal_before = paper_exchange.get_balance()['free']['USDT']

        paper_exchange.place_order('BTCUSDT', 'sell', 0.09)
        bal_after = paper_exchange.get_balance()['free']['USDT']
        assert bal_after > bal_before
        assert paper_exchange.get_balance()['free']['BTC'] == pytest.approx(0.10, abs=1e-8)

    def test_slippage_applied_on_buy(self, paper_exchange, monkeypatch):
        """Fill price should be slightly above market price for buys."""
        monkeypatch.setattr(paper_exchange, 'get_ticker', lambda sym: {'last': 50_000.0})
        order = paper_exchange.place_order('BTCUSDT', 'buy', 0.01)
        expected_max = 50_000.0 * (1 + paper_exchange.SLIPPAGE)
        assert order['price'] <= expected_max + 1.0

    def test_slippage_applied_on_sell(self, paper_exchange, monkeypatch):
        """Fill price should be slightly below market price for sells."""
        monkeypatch.setattr(paper_exchange, 'get_ticker', lambda sym: {'last': 50_000.0})
        paper_exchange._paper_balance['BTC'] = 1.0
        order = paper_exchange.place_order('BTCUSDT', 'sell', 0.1)
        expected_min = 50_000.0 * (1 - paper_exchange.SLIPPAGE)
        assert order['price'] >= expected_min - 1.0

    def test_insufficient_balance_raises(self, paper_exchange, monkeypatch):
        """Buying more than balance should raise ValueError."""
        monkeypatch.setattr(paper_exchange, 'get_ticker', lambda sym: {'last': 50_000.0})
        with pytest.raises(ValueError, match='Insufficient'):
            paper_exchange.place_order('BTCUSDT', 'buy', 100.0)

    def test_get_open_orders_empty(self, paper_exchange):
        orders = paper_exchange.get_open_orders('BTCUSDT')
        assert isinstance(orders, list)
        assert len(orders) == 0

    def test_cancel_order(self, paper_exchange, monkeypatch):
        """Cancelling a known order ID should return True."""
        monkeypatch.setattr(paper_exchange, 'get_ticker', lambda sym: {'last': 50_000.0})
        # Manually add a paper order to the list
        paper_exchange._paper_orders.append({
            'id': 'test-order-001',
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'status': 'open',
        })
        result = paper_exchange.cancel_order('test-order-001', 'BTCUSDT')
        assert result is True
        assert len(paper_exchange._paper_orders) == 0

    def test_cancel_nonexistent_order(self, paper_exchange):
        result = paper_exchange.cancel_order('nonexistent-id', 'BTCUSDT')
        assert result is False

    def test_order_status_returns_closed(self, paper_exchange):
        """Order that was never tracked returns 'closed'."""
        status = paper_exchange.get_order_status('unknown-id', 'BTCUSDT')
        assert status['status'] == 'closed'

    def test_no_api_keys_required_in_paper_mode(self):
        """Paper mode should initialise without API keys."""
        from exchange_connector import ExchangeConnector
        ex = ExchangeConnector(paper_mode=True, initial_balance=5_000)
        assert ex.paper_mode is True
        assert ex.get_balance()['free']['USDT'] == pytest.approx(5_000.0)

    def test_retry_with_backoff_raises_after_retries(self, paper_exchange):
        """_retry_with_backoff should raise RuntimeError after max_retries."""
        def always_fail(*a, **kw):
            raise ConnectionError("forced failure")
        with pytest.raises(RuntimeError, match='retries failed'):
            paper_exchange._retry_with_backoff(always_fail, max_retries=2)
