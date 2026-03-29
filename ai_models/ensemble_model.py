"""Ensemble model combining LSTM, XGBoost, and Random Forest predictions."""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from ai_models.feature_engineering import FeatureEngineer
from ai_models.lstm_model import LSTMModel
from ai_models.xgboost_model import XGBoostModel
from ai_models.random_forest_model import RandomForestModel


class EnsembleModel:
    """
    Weighted ensemble of LSTM (40%), XGBoost (35%), RandomForest (25%).

    Produces a final BUY/SELL/HOLD signal when ensemble confidence >= 0.65.
    """

    DEFAULT_WEIGHTS = {'lstm': 0.40, 'xgboost': 0.35, 'random_forest': 0.25}
    MIN_CONFIDENCE = 0.65

    def __init__(self, symbol: str, models_dir: str = './saved_models'):
        self.symbol = symbol
        self.models_dir = models_dir
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.feature_engineer = FeatureEngineer(symbol)
        self.lstm = LSTMModel(symbol, models_dir)
        self.xgb = XGBoostModel(symbol, models_dir)
        self.rf = RandomForestModel(symbol, models_dir)
        self._models_loaded = False

    def load_models(self) -> None:
        """Load all sub-models from saved_models directory."""
        errors = []
        try:
            self.lstm.load()
        except Exception as e:
            errors.append(f'LSTM load failed: {e}')
        try:
            self.xgb.load()
        except Exception as e:
            errors.append(f'XGBoost load failed: {e}')
        try:
            self.rf.load()
        except Exception as e:
            errors.append(f'RF load failed: {e}')
        if errors:
            import warnings
            warnings.warn('; '.join(errors))
        self._models_loaded = True

    def predict(
        self, ohlcv_df: pd.DataFrame
    ) -> Tuple[str, float, Dict]:
        """
        Compute ensemble prediction from raw OHLCV data.

        Returns:
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: float in [0, 1]
            details: per-model probabilities and signals
        """
        df = self.feature_engineer.create_features(ohlcv_df)

        # --- tabular features for XGBoost / RF ---
        X_tab, _ = self.feature_engineer.prepare_tabular(df)
        probs_xgb = self.xgb.predict(X_tab) if self.xgb.model else np.array([[1/3, 1/3, 1/3]])
        probs_rf = self.rf.predict(X_tab) if self.rf.rf else np.array([[1/3, 1/3, 1/3]])

        # --- sequences for LSTM ---
        try:
            X_seq, _ = self.feature_engineer.prepare_sequences(df)
            probs_lstm = self.lstm.predict(X_seq) if self.lstm.model else np.array([[1/3, 1/3, 1/3]])
        except Exception:
            probs_lstm = np.array([[1/3, 1/3, 1/3]])

        # Align lengths to minimum
        n = min(len(probs_lstm), len(probs_xgb), len(probs_rf))
        probs_lstm = probs_lstm[-n:]
        probs_xgb = probs_xgb[-n:]
        probs_rf = probs_rf[-n:]

        ensemble_probs = self._weighted_vote(probs_lstm, probs_xgb, probs_rf)
        last = ensemble_probs[-1]
        idx = int(np.argmax(last))
        conf = float(last[idx])
        label_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        signal = label_map[idx] if conf >= self.MIN_CONFIDENCE else 'HOLD'

        details = {
            'ensemble_probs': last.tolist(),
            'lstm_probs': probs_lstm[-1].tolist(),
            'xgb_probs': probs_xgb[-1].tolist(),
            'rf_probs': probs_rf[-1].tolist(),
            'weights': dict(self.weights),
        }
        return signal, conf, details

    def _weighted_vote(
        self,
        probs_lstm: np.ndarray,
        probs_xgb: np.ndarray,
        probs_rf: np.ndarray,
    ) -> np.ndarray:
        """Compute weighted average of probability matrices."""
        w = self.weights
        return (
            w['lstm'] * probs_lstm
            + w['xgboost'] * probs_xgb
            + w['random_forest'] * probs_rf
        )

    def auto_weight_models(self, validation_results: Dict) -> None:
        """
        Adjust ensemble weights based on validation accuracies.

        Weights are softmax-normalised from accuracy scores so they sum to 1.
        """
        accs = {
            'lstm': validation_results.get('lstm', {}).get('accuracy', 1/3),
            'xgboost': validation_results.get('xgboost', {}).get('accuracy', 1/3),
            'random_forest': validation_results.get('random_forest', {}).get('accuracy', 1/3),
        }
        total = sum(accs.values()) or 1.0
        self.weights = {k: v / total for k, v in accs.items()}

    def get_signal(
        self, ohlcv_df: pd.DataFrame
    ) -> Tuple[str, float, Dict]:
        """Alias for predict — returns (signal, confidence, details)."""
        return self.predict(ohlcv_df)

    def explain_decision(self, ohlcv_df: pd.DataFrame) -> str:
        """Return human-readable explanation of the latest ensemble decision."""
        signal, confidence, details = self.predict(ohlcv_df)
        ep = details['ensemble_probs']
        lines = [
            f"=== Ensemble Decision for {self.symbol} ===",
            f"Signal     : {signal}",
            f"Confidence : {confidence:.2%}",
            f"",
            f"Ensemble probabilities  → HOLD={ep[0]:.2%}  BUY={ep[1]:.2%}  SELL={ep[2]:.2%}",
            f"",
            f"Individual model contributions (weights: LSTM={self.weights['lstm']:.0%}, "
            f"XGB={self.weights['xgboost']:.0%}, RF={self.weights['random_forest']:.0%}):",
            f"  LSTM        → HOLD={details['lstm_probs'][0]:.2%}  BUY={details['lstm_probs'][1]:.2%}  SELL={details['lstm_probs'][2]:.2%}",
            f"  XGBoost     → HOLD={details['xgb_probs'][0]:.2%}  BUY={details['xgb_probs'][1]:.2%}  SELL={details['xgb_probs'][2]:.2%}",
            f"  RandomForest→ HOLD={details['rf_probs'][0]:.2%}  BUY={details['rf_probs'][1]:.2%}  SELL={details['rf_probs'][2]:.2%}",
            f"",
            f"Min confidence threshold: {self.MIN_CONFIDENCE:.0%}",
        ]
        return '\n'.join(lines)
