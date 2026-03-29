"""Model trainer: data fetching, dataset preparation, and full training pipeline."""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ai_models.feature_engineering import FeatureEngineer
from ai_models.lstm_model import LSTMModel
from ai_models.xgboost_model import XGBoostModel
from ai_models.random_forest_model import RandomForestModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates data fetching, feature engineering, and training of all models.
    """

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    # TEST_RATIO = 0.15

    def __init__(
        self,
        symbol: str,
        models_dir: str = './saved_models',
        logs_dir: str = './logs',
    ):
        self.symbol = symbol
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        self.feature_engineer = FeatureEngineer(symbol)
        self.lstm = LSTMModel(symbol, models_dir)
        self.xgb = XGBoostModel(symbol, models_dir)
        self.rf = RandomForestModel(symbol, models_dir)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_historical_data(
        self,
        start_date: str,
        end_date: str,
        timeframe: str = '15m',
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data via ccxt Binance with pagination.

        Args:
            start_date: ISO date string, e.g. '2020-01-01'
            end_date:   ISO date string, e.g. '2024-01-01'
            timeframe:  ccxt timeframe string, default '15m'

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            import ccxt
        except ImportError as e:
            raise ImportError("ccxt is required for data fetching.") from e

        from config.settings import ASSETS_CONFIG
        ccxt_symbol = ASSETS_CONFIG.get(self.symbol, {}).get('ccxt_symbol', 'BTC/USDT')

        exchange = ccxt.binance({'enableRateLimit': True})
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        timeframe_ms = {
            '1m': 60_000, '5m': 300_000, '15m': 900_000,
            '30m': 1_800_000, '1h': 3_600_000, '4h': 14_400_000,
        }.get(timeframe, 900_000)

        limit = 1000
        all_ohlcv: List[list] = []
        current_ts = start_ts

        logger.info(f"Fetching {ccxt_symbol} {timeframe} from {start_date} to {end_date}")
        with tqdm(desc='Fetching OHLCV', unit='batch') as pbar:
            while current_ts < end_ts:
                try:
                    batch = exchange.fetch_ohlcv(
                        ccxt_symbol, timeframe, since=current_ts, limit=limit
                    )
                except Exception as e:
                    logger.warning(f"Fetch error: {e}, retrying after 5s")
                    import time
                    time.sleep(5)
                    continue

                if not batch:
                    break
                all_ohlcv.extend(batch)
                current_ts = batch[-1][0] + timeframe_ms
                pbar.update(1)
                if current_ts >= end_ts:
                    break

        if not all_ohlcv:
            raise ValueError(f"No data fetched for {ccxt_symbol} between {start_date} and {end_date}")

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        # Clip to requested range
        df = df[df.index < pd.Timestamp(end_date, tz='UTC')]
        return df

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Build feature-engineered train/val/test splits (70/15/15).

        Returns dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
        X_seq_train, X_seq_val, X_seq_test, y_seq_train, y_seq_val, y_seq_test
        """
        df_feat = self.feature_engineer.create_features(df)
        self.feature_engineer.fit_scaler(df_feat)
        df_scaled = self.feature_engineer.transform(df_feat)

        X_tab, y_tab = self.feature_engineer.prepare_tabular(df_scaled)
        X_seq, y_seq = self.feature_engineer.prepare_sequences(df_scaled)

        n_tab = len(X_tab)
        n_seq = len(X_seq)

        t1_tab = int(n_tab * self.TRAIN_RATIO)
        t2_tab = int(n_tab * (self.TRAIN_RATIO + self.VAL_RATIO))
        t1_seq = int(n_seq * self.TRAIN_RATIO)
        t2_seq = int(n_seq * (self.TRAIN_RATIO + self.VAL_RATIO))

        return {
            'X_train': X_tab[:t1_tab],
            'y_train': y_tab[:t1_tab],
            'X_val':   X_tab[t1_tab:t2_tab],
            'y_val':   y_tab[t1_tab:t2_tab],
            'X_test':  X_tab[t2_tab:],
            'y_test':  y_tab[t2_tab:],
            'X_seq_train': X_seq[:t1_seq],
            'y_seq_train': y_seq[:t1_seq],
            'X_seq_val':   X_seq[t1_seq:t2_seq],
            'y_seq_val':   y_seq[t1_seq:t2_seq],
            'X_seq_test':  X_seq[t2_seq:],
            'y_seq_test':  y_seq[t2_seq:],
            'feature_names': self.feature_engineer.get_feature_names(),
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_all_models(
        self,
        df: Optional[pd.DataFrame] = None,
        start_date: str = '2020-01-01',
        end_date: str = '2024-01-01',
    ) -> Dict:
        """Fetch data (if not provided), prepare dataset, train all three models."""
        if df is None:
            df = self.fetch_historical_data(start_date, end_date)

        dataset = self.prepare_dataset(df)
        metrics = {}

        logger.info("Training XGBoost...")
        self.xgb.train(
            dataset['X_train'], dataset['y_train'],
            dataset['X_val'], dataset['y_val'],
            feature_names=dataset['feature_names'],
        )
        metrics['xgboost'] = self.xgb.evaluate(dataset['X_test'], dataset['y_test'])
        self.xgb.save()

        logger.info("Training RandomForest...")
        self.rf.train(
            dataset['X_train'], dataset['y_train'],
            feature_names=dataset['feature_names'],
        )
        metrics['random_forest'] = self.rf.evaluate(dataset['X_test'], dataset['y_test'])
        self.rf.save()

        logger.info("Training LSTM...")
        try:
            self.lstm.train(
                dataset['X_seq_train'], dataset['y_seq_train'],
                dataset['X_seq_val'], dataset['y_seq_val'],
            )
            metrics['lstm'] = self.lstm.evaluate(dataset['X_seq_test'], dataset['y_seq_test'])
            self.lstm.save()
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
            metrics['lstm'] = {'error': str(e)}

        self.save_training_report(metrics)
        return metrics

    def evaluate_all_models(self, dataset: Dict) -> Dict:
        """Run evaluation for all three models on the provided dataset splits."""
        return {
            'xgboost': self.xgb.evaluate(dataset['X_test'], dataset['y_test']),
            'random_forest': self.rf.evaluate(dataset['X_test'], dataset['y_test']),
            'lstm': self.lstm.evaluate(dataset['X_seq_test'], dataset['y_seq_test']),
        }

    def save_training_report(self, metrics: Dict) -> None:
        """Persist training metrics to a JSON report file."""
        report = {
            'symbol': self.symbol,
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'metrics': metrics,
        }
        path = os.path.join(self.logs_dir, f'training_report_{self.symbol}.json')
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Training report saved to {path}")

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def run_walk_forward_validation(
        self, df: pd.DataFrame, n_splits: int = 5
    ) -> Dict:
        """
        Perform expanding-window walk-forward validation.

        Returns dict with per-fold metrics and aggregated summary.
        """
        df_feat = self.feature_engineer.create_features(df)
        self.feature_engineer.fit_scaler(df_feat)
        df_scaled = self.feature_engineer.transform(df_feat)
        X_tab, y_tab = self.feature_engineer.prepare_tabular(df_scaled)

        fold_size = len(X_tab) // (n_splits + 1)
        fold_metrics = []

        for i in range(1, n_splits + 1):
            train_end = fold_size * i
            test_end = train_end + fold_size
            X_tr, y_tr = X_tab[:train_end], y_tab[:train_end]
            X_te, y_te = X_tab[train_end:test_end], y_tab[train_end:test_end]
            if len(np.unique(y_tr)) < 2:
                continue
            rf_tmp = RandomForestModel(self.symbol, self.models_dir)
            rf_tmp.train(X_tr, y_tr)
            fold_metrics.append(rf_tmp.evaluate(X_te, y_te))

        mean_acc = float(np.mean([m['accuracy'] for m in fold_metrics])) if fold_metrics else 0.0
        return {'fold_metrics': fold_metrics, 'mean_accuracy': mean_acc, 'n_splits': n_splits}

    def retrain_on_new_data(self, new_data: pd.DataFrame) -> Dict:
        """Incremental retrain on recently arrived data."""
        logger.info(f"Retraining {self.symbol} on {len(new_data)} new rows")
        return self.train_all_models(df=new_data)
