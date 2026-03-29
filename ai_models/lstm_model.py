"""LSTM model for sequence-based trading signal prediction."""

import os
import numpy as np
from typing import Dict, Tuple, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


class LSTMModel:
    """Stacked LSTM classifier for BUY/SELL/HOLD prediction."""

    def __init__(self, symbol: str, models_dir: str = './saved_models'):
        self.symbol = symbol
        self.models_dir = models_dir
        self.model: Optional[object] = None
        self.history = None
        os.makedirs(models_dir, exist_ok=True)

    def build_model(self, n_features: int, sequence_length: int = 60) -> None:
        """
        Build stacked LSTM network.

        Architecture: LSTM(128) → LSTM(64) → LSTM(32) → Dense(64) → Dense(32) → Dense(3)
        """
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMModel.")

        model = keras.Sequential([
            keras.layers.Input(shape=(sequence_length, n_features)),
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='softmax'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        self.model = model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> dict:
        """
        Train model with early stopping and LR reduction.

        Returns:
            Training history as dict.
        """
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow is required.")
        if self.model is None:
            n_features = X_train.shape[2]
            seq_len = X_train.shape[1]
            self.build_model(n_features, seq_len)

        checkpoint_path = os.path.join(self.models_dir, f'lstm_{self.symbol}_best.keras')
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0,
            ),
        ]
        hist = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        self.history = hist.history
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class probability array of shape (n, 3)."""
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")
        return self.model.predict(X, verbose=0)

    def get_signal(self, X: np.ndarray) -> Tuple[str, float]:
        """
        Return (signal, confidence) for the last sample in X.

        Only returns BUY/SELL if confidence > 0.55, otherwise HOLD.
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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Compute accuracy and loss on test set."""
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow required.")
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        probs = self.predict(X_test)
        preds = np.argmax(probs, axis=1)
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        return {
            'loss': float(loss),
            'accuracy': float(acc),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save model weights to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_path = path or os.path.join(self.models_dir, f'lstm_{self.symbol}.keras')
        self.model.save(save_path)

    def load(self, path: Optional[str] = None) -> None:
        """Load model from disk."""
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow required.")
        load_path = path or os.path.join(self.models_dir, f'lstm_{self.symbol}.keras')
        self.model = keras.models.load_model(load_path)
