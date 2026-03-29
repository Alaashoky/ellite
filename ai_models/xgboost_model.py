"""XGBoost classifier for trading signal prediction."""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix


class XGBoostModel:
    """XGBoost-based BUY/SELL/HOLD classifier with hyperparameter tuning."""

    def __init__(self, symbol: str, models_dir: str = './saved_models'):
        self.symbol = symbol
        self.models_dir = models_dir
        self.model: Optional[XGBClassifier] = None
        self.feature_names: List[str] = []
        os.makedirs(models_dir, exist_ok=True)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> XGBClassifier:
        """
        Train XGBoost classifier with early stopping.

        Args:
            X_train: Training features
            y_train: Training labels (0/1/2)
            X_val:   Validation features
            y_val:   Validation labels
            feature_names: Optional list of feature names
        """
        if feature_names:
            self.feature_names = feature_names

        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            early_stopping_rounds=50,
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        return self.model

    def hyperparameter_tuning(
        self, X: np.ndarray, y: np.ndarray, n_iter: int = 20
    ) -> XGBClassifier:
        """
        Randomized search over hyperparameter space.

        Returns best estimator fitted on full data.
        """
        param_dist = {
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
            'n_estimators': [200, 300, 400, 500],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0.0, 0.05, 0.1, 0.2],
        }
        base = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
        search = RandomizedSearchCV(
            base, param_dist, n_iter=n_iter, cv=3,
            scoring='accuracy', verbose=1, random_state=42, n_jobs=-1,
        )
        search.fit(X, y)
        self.model = search.best_estimator_
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class probability array of shape (n, 3)."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        return self.model.predict_proba(X)

    def get_signal(self, X: np.ndarray) -> Tuple[str, float]:
        """
        Return (signal, confidence) for the latest sample.

        Only BUY/SELL when confidence > 0.55.
        """
        probs = self.predict(X)
        last = probs[-1]
        idx = int(np.argmax(last))
        conf = float(last[idx])
        label_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        signal = label_map[idx]
        if signal in ('BUY', 'SELL') and conf <= 0.55:
            signal = 'HOLD'
        return signal, conf

    def get_feature_importance(self) -> pd.DataFrame:
        """Return sorted feature importance DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        importances = self.model.feature_importances_
        names = self.feature_names if self.feature_names else [f'f{i}' for i in range(len(importances))]
        df = pd.DataFrame({'feature': names, 'importance': importances})
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate on held-out test set."""
        probs = self.predict(X_test)
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y_test))
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        return {
            'accuracy': acc,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
        }

    def save(self, path: Optional[str] = None) -> None:
        """Persist model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_path = path or os.path.join(self.models_dir, f'xgb_{self.symbol}.joblib')
        joblib.dump({'model': self.model, 'feature_names': self.feature_names}, save_path)

    def load(self, path: Optional[str] = None) -> None:
        """Load model from disk."""
        load_path = path or os.path.join(self.models_dir, f'xgb_{self.symbol}.joblib')
        data = joblib.load(load_path)
        self.model = data['model']
        self.feature_names = data.get('feature_names', [])
