"""Random Forest + Gradient Boosting ensemble model for trading signals."""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


class RandomForestModel:
    """
    Combined RF + GradientBoosting model.

    Final probabilities = 60% RF + 40% GradientBoosting.
    """

    RF_WEIGHT = 0.60
    GB_WEIGHT = 0.40

    def __init__(self, symbol: str, models_dir: str = './saved_models'):
        self.symbol = symbol
        self.models_dir = models_dir
        self.rf: Optional[RandomForestClassifier] = None
        self.gb: Optional[GradientBoostingClassifier] = None
        self.feature_names: List[str] = []
        os.makedirs(models_dir, exist_ok=True)

    def _build_rf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )

    def _build_gb(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            random_state=42,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit RF and GB on training data."""
        if feature_names:
            self.feature_names = feature_names
        self.rf = self._build_rf()
        self.gb = self._build_gb()
        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return weighted-average probability array of shape (n, 3)."""
        if self.rf is None or self.gb is None:
            raise RuntimeError("Model not trained or loaded.")
        rf_probs = self.rf.predict_proba(X)
        gb_probs = self.gb.predict_proba(X)
        return self.RF_WEIGHT * rf_probs + self.GB_WEIGHT * gb_probs

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
        """Return RF feature importances sorted descending."""
        if self.rf is None:
            raise RuntimeError("Model not trained or loaded.")
        names = self.feature_names if self.feature_names else [f'f{i}' for i in range(len(self.rf.feature_importances_))]
        df = pd.DataFrame({'feature': names, 'importance': self.rf.feature_importances_})
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """Stratified k-fold cross-validation of the RF component."""
        rf = self._build_rf()
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        return {
            'mean_accuracy': float(scores.mean()),
            'std_accuracy': float(scores.std()),
            'fold_scores': scores.tolist(),
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate combined model on test set."""
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
        """Persist both models to disk."""
        if self.rf is None:
            raise RuntimeError("No model to save.")
        save_path = path or os.path.join(self.models_dir, f'rf_{self.symbol}.joblib')
        joblib.dump({
            'rf': self.rf,
            'gb': self.gb,
            'feature_names': self.feature_names,
        }, save_path)

    def load(self, path: Optional[str] = None) -> None:
        """Load models from disk."""
        load_path = path or os.path.join(self.models_dir, f'rf_{self.symbol}.joblib')
        data = joblib.load(load_path)
        self.rf = data['rf']
        self.gb = data['gb']
        self.feature_names = data.get('feature_names', [])
