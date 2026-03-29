"""Unit tests for AI models: FeatureEngineer, LSTMModel, XGBoostModel,
RandomForestModel, and RiskManager."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, symbol: str = 'BTCUSDT') -> pd.DataFrame:
    """Generate synthetic OHLCV data with a DatetimeIndex."""
    rng = np.random.default_rng(42)
    base = 30_000.0 if 'BTC' in symbol else 1_900.0
    close = base + np.cumsum(rng.normal(0, base * 0.005, n))
    close = np.abs(close)
    noise = rng.uniform(0.001, 0.005, n) * close
    df = pd.DataFrame({
        'open':   close - noise,
        'high':   close + noise * 1.5,
        'low':    close - noise * 1.5,
        'close':  close,
        'volume': rng.uniform(1, 100, n),
    }, index=pd.date_range('2023-01-01', periods=n, freq='15min', tz='UTC'))
    return df


# ---------------------------------------------------------------------------
# TestFeatureEngineer
# ---------------------------------------------------------------------------

class TestFeatureEngineer:
    def test_btc_features_shape(self):
        from ai_models.feature_engineering import FeatureEngineer
        fe = FeatureEngineer('BTCUSDT')
        df = _make_ohlcv(300)
        result = fe.create_features(df)
        assert len(result) == 300
        assert 'rsi_14' in result.columns
        assert 'macd' in result.columns
        assert 'ema_50' in result.columns
        assert 'atr_14' in result.columns
        assert 'bb_upper' in result.columns

    def test_gold_features_shape(self):
        from ai_models.feature_engineering import FeatureEngineer
        fe = FeatureEngineer('XAUUSDT')
        df = _make_ohlcv(300, 'XAUUSDT')
        result = fe.create_features(df)
        assert len(result) == 300
        assert fe.is_gold
        assert fe.label_min_return == 0.001

    def test_btc_thresholds(self):
        from ai_models.feature_engineering import FeatureEngineer
        fe = FeatureEngineer('BTCUSDT')
        assert fe.is_btc
        assert fe.label_min_return == 0.003

    def test_labels_validity(self):
        from ai_models.feature_engineering import FeatureEngineer
        fe = FeatureEngineer('BTCUSDT')
        df = _make_ohlcv(300)
        labels = fe.generate_labels(df, forward_periods=5)
        assert set(labels.unique()).issubset({0, 1, 2})
        assert len(labels) == 300

    def test_sequence_shape(self):
        from ai_models.feature_engineering import FeatureEngineer
        fe = FeatureEngineer('BTCUSDT')
        df = _make_ohlcv(300)
        df_feat = fe.create_features(df)
        X, y = fe.prepare_sequences(df_feat, sequence_length=60)
        assert X.ndim == 3
        assert X.shape[1] == 60
        assert len(X) == len(y)

    def test_tabular_shape(self):
        from ai_models.feature_engineering import FeatureEngineer
        fe = FeatureEngineer('BTCUSDT')
        df = _make_ohlcv(300)
        df_feat = fe.create_features(df)
        X, y = fe.prepare_tabular(df_feat)
        assert X.ndim == 2
        assert len(X) == len(y)

    def test_no_inf_values(self):
        from ai_models.feature_engineering import FeatureEngineer
        fe = FeatureEngineer('BTCUSDT')
        df = _make_ohlcv(300)
        result = fe.create_features(df)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.any(np.isinf(numeric.values)), "Inf values found in features"

    def test_scaler_fit_transform(self):
        from ai_models.feature_engineering import FeatureEngineer
        fe = FeatureEngineer('BTCUSDT')
        df = _make_ohlcv(300)
        df_feat = fe.create_features(df)
        fe.fit_scaler(df_feat)
        df_scaled = fe.transform(df_feat)
        assert df_scaled is not None
        assert len(df_scaled) == len(df_feat)


# ---------------------------------------------------------------------------
# TestLSTMModel
# ---------------------------------------------------------------------------

class TestLSTMModel:
    def test_build_model_creates_model(self):
        pytest.importorskip('tensorflow')
        from ai_models.lstm_model import LSTMModel
        model = LSTMModel('BTCUSDT', models_dir='./test_models')
        model.build_model(n_features=20, sequence_length=30)
        assert model.model is not None

    def test_predict_shape(self):
        pytest.importorskip('tensorflow')
        from ai_models.lstm_model import LSTMModel
        model = LSTMModel('BTCUSDT', models_dir='./test_models')
        model.build_model(n_features=10, sequence_length=20)
        X = np.random.rand(5, 20, 10).astype(np.float32)
        probs = model.predict(X)
        assert probs.shape == (5, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_get_signal_returns_valid(self):
        pytest.importorskip('tensorflow')
        from ai_models.lstm_model import LSTMModel
        model = LSTMModel('BTCUSDT', models_dir='./test_models')
        model.build_model(n_features=10, sequence_length=20)
        X = np.random.rand(3, 20, 10).astype(np.float32)
        signal, conf = model.get_signal(X)
        assert signal in ('BUY', 'SELL', 'HOLD')
        assert 0.0 <= conf <= 1.0

    def test_no_tensorflow_raises(self):
        with patch.dict('sys.modules', {'tensorflow': None}):
            from importlib import reload
            import ai_models.lstm_model as lm_module
            reload(lm_module)
            m = lm_module.LSTMModel('BTCUSDT')
            with pytest.raises(ImportError):
                m.build_model(5)


# ---------------------------------------------------------------------------
# TestXGBoostModel
# ---------------------------------------------------------------------------

class TestXGBoostModel:
    def _make_dataset(self, n=200, n_features=20):
        X = np.random.rand(n, n_features).astype(np.float32)
        y = np.random.randint(0, 3, n).astype(np.int32)
        return X, y

    def test_train_and_predict(self):
        from ai_models.xgboost_model import XGBoostModel
        model = XGBoostModel('BTCUSDT', models_dir='./test_models')
        X, y = self._make_dataset()
        X_tr, y_tr = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]
        model.train(X_tr, y_tr, X_val, y_val)
        probs = model.predict(X_val)
        assert probs.shape == (len(X_val), 3)

    def test_get_signal(self):
        from ai_models.xgboost_model import XGBoostModel
        model = XGBoostModel('BTCUSDT', models_dir='./test_models')
        X, y = self._make_dataset()
        model.train(X[:150], y[:150], X[150:], y[150:])
        signal, conf = model.get_signal(X[150:])
        assert signal in ('BUY', 'SELL', 'HOLD')
        assert 0.0 <= conf <= 1.0

    def test_feature_importance(self):
        from ai_models.xgboost_model import XGBoostModel
        model = XGBoostModel('BTCUSDT', models_dir='./test_models')
        X, y = self._make_dataset(n_features=5)
        model.train(X[:150], y[:150], X[150:], y[150:], feature_names=['a','b','c','d','e'])
        fi = model.get_feature_importance()
        assert 'feature' in fi.columns
        assert 'importance' in fi.columns
        assert len(fi) == 5


# ---------------------------------------------------------------------------
# TestRandomForestModel
# ---------------------------------------------------------------------------

class TestRandomForestModel:
    def _make_dataset(self, n=200, n_features=20):
        X = np.random.rand(n, n_features).astype(np.float32)
        y = np.random.randint(0, 3, n).astype(np.int32)
        return X, y

    def test_train_predict(self):
        from ai_models.random_forest_model import RandomForestModel
        model = RandomForestModel('BTCUSDT', models_dir='./test_models')
        X, y = self._make_dataset()
        model.train(X[:160], y[:160])
        probs = model.predict(X[160:])
        assert probs.shape[1] == 3

    def test_get_signal(self):
        from ai_models.random_forest_model import RandomForestModel
        model = RandomForestModel('BTCUSDT', models_dir='./test_models')
        X, y = self._make_dataset()
        model.train(X[:160], y[:160])
        signal, conf = model.get_signal(X[160:])
        assert signal in ('BUY', 'SELL', 'HOLD')

    def test_evaluate_keys(self):
        from ai_models.random_forest_model import RandomForestModel
        model = RandomForestModel('BTCUSDT', models_dir='./test_models')
        X, y = self._make_dataset()
        model.train(X[:160], y[:160])
        result = model.evaluate(X[160:], y[160:])
        assert 'accuracy' in result
        assert 'confusion_matrix' in result


# ---------------------------------------------------------------------------
# TestRiskManager
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_position_size_basic(self):
        from risk_manager import RiskManager
        rm = RiskManager(initial_balance=10_000, risk_per_trade=0.01)
        # 1% of 10000 = 100 USDT risk; price risk = 500
        size = rm.calculate_position_size(entry_price=50_000, stop_loss=49_500)
        assert size > 0
        assert abs(size - 0.2) < 0.01

    def test_position_size_zero_on_equal_prices(self):
        from risk_manager import RiskManager
        rm = RiskManager(initial_balance=10_000)
        size = rm.calculate_position_size(entry_price=100.0, stop_loss=100.0)
        assert size == 0.0

    def test_daily_risk_limit_ok(self):
        from risk_manager import RiskManager
        rm = RiskManager(initial_balance=10_000, max_daily_risk=0.05)
        assert rm.check_daily_risk_limit() is True

    def test_daily_risk_limit_breached(self):
        from risk_manager import RiskManager
        rm = RiskManager(initial_balance=10_000, max_daily_risk=0.05)
        rm.update_balance(-600)  # > 5% loss
        assert rm.check_daily_risk_limit() is False

    def test_drawdown_check_ok(self):
        from risk_manager import RiskManager
        rm = RiskManager(initial_balance=10_000, max_drawdown=0.15)
        assert rm.check_max_drawdown() is True

    def test_drawdown_check_breached(self):
        from risk_manager import RiskManager
        rm = RiskManager(initial_balance=10_000, max_drawdown=0.15)
        rm.update_balance(-2_000)  # 20% drawdown
        assert rm.check_max_drawdown() is False

    def test_validate_trade_low_confidence(self):
        from risk_manager import RiskManager
        rm = RiskManager(initial_balance=10_000)
        allowed, reason = rm.validate_trade('BUY', confidence=0.3)
        assert allowed is False
        assert 'confidence' in reason.lower()

    def test_max_positions(self):
        from risk_manager import RiskManager
        rm = RiskManager(initial_balance=10_000)
        rm.add_position('BTCUSDT', {})
        rm.add_position('XAUUSDT', {})
        rm.add_position('ETHUSDT', {})
        assert rm.check_max_concurrent_positions(max_positions=3) is False

    def test_get_risk_summary_keys(self):
        from risk_manager import RiskManager
        rm = RiskManager(initial_balance=10_000)
        summary = rm.get_risk_summary()
        for key in ('current_balance', 'peak_balance', 'current_drawdown', 'daily_pnl'):
            assert key in summary
