"""Feature engineering for AI trading models."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import joblib
import os


class FeatureEngineer:
    """Generates technical and statistical features for ML models."""

    def __init__(self, symbol: str):
        """
        Initialize FeatureEngineer for a specific trading symbol.

        Args:
            symbol: Trading symbol, e.g. 'BTCUSDT' or 'XAUUSDT'
        """
        self.symbol = symbol
        self.is_btc = symbol == 'BTCUSDT'
        self.is_gold = symbol == 'XAUUSDT'
        self.label_min_return = 0.003 if self.is_btc else 0.001
        self.scaler = StandardScaler()
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Master method: apply all feature groups and return enriched df."""
        df = df.copy()
        df = self._add_technical_indicators(df)
        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        df = self._add_market_structure_features(df)
        df = self._add_session_features(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def generate_labels(self, df: pd.DataFrame, forward_periods: int = 5) -> pd.Series:
        """
        Generate 3-class labels: 0=HOLD, 1=BUY, 2=SELL.

        Uses forward returns over `forward_periods` candles.
        BTC threshold: 0.3%, Gold threshold: 0.1%.
        """
        fwd_return = (df['close'].shift(-forward_periods) - df['close']) / df['close']
        threshold = self.label_min_return
        labels = pd.Series(0, index=df.index, name='label')
        labels[fwd_return >= threshold] = 1
        labels[fwd_return <= -threshold] = 2
        return labels

    def prepare_sequences(
        self, df: pd.DataFrame, sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create LSTM sequences of shape (n_samples, sequence_length, n_features)."""
        feature_cols = self._get_numeric_feature_cols(df)
        labels = self.generate_labels(df)
        X_arr = df[feature_cols].values
        y_arr = labels.values
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_arr)):
            X_seq.append(X_arr[i - sequence_length:i])
            y_seq.append(y_arr[i])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32)

    def prepare_tabular(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare flat feature matrix for XGBoost/RF."""
        feature_cols = self._get_numeric_feature_cols(df)
        labels = self.generate_labels(df)
        X = df[feature_cols].values.astype(np.float32)
        y = labels.values.astype(np.int32)
        return X, y

    def get_feature_names(self) -> List[str]:
        """Return list of feature column names."""
        return list(self._feature_names)

    def fit_scaler(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit StandardScaler on numeric feature columns."""
        feature_cols = self._get_numeric_feature_cols(df)
        self._feature_names = feature_cols
        self.scaler.fit(df[feature_cols].values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to feature columns."""
        feature_cols = self._get_numeric_feature_cols(df)
        df = df.copy()
        df[feature_cols] = self.scaler.transform(df[feature_cols].values)
        return df

    def save_scaler(self, path: str) -> None:
        """Persist scaler to disk."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str) -> None:
        """Load scaler from disk."""
        self.scaler = joblib.load(path)

    # ------------------------------------------------------------------
    # Internal feature builders
    # ------------------------------------------------------------------

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, MACD, EMA, ATR, Bollinger Bands, Stochastic, ADX, CCI, Williams %R."""
        close = df['close']
        high = df['high']
        low = df['low']

        # --- RSI(14) ---
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # --- MACD(12,26,9) ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # --- EMA(9,21,50,200) ---
        for span in [9, 21, 50, 200]:
            df[f'ema_{span}'] = close.ewm(span=span, adjust=False).mean()

        # --- ATR(14) ---
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = true_range.ewm(com=13, adjust=False).mean()

        # --- Bollinger Bands(20,2) ---
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_mid'] = bb_mid
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid
        df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # --- Stochastic(14,3) ---
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        df['stoch_k'] = (close - low14) / (high14 - low14 + 1e-10) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # --- ADX(14) ---
        df['adx_14'] = self._calc_adx(high, low, close, 14)

        # --- CCI(20) ---
        tp = (high + low + close) / 3
        tp_mean = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['cci_20'] = (tp - tp_mean) / (0.015 * tp_mad + 1e-10)

        # --- Williams %R(14) ---
        hh14 = high.rolling(14).max()
        ll14 = low.rolling(14).min()
        df['willr_14'] = (hh14 - close) / (hh14 - ll14 + 1e-10) * -100

        return df

    def _calc_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate ADX without lookahead bias."""
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        plus_dm = high - prev_high
        minus_dm = prev_low - low
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        atr = tr.ewm(com=period - 1, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / (atr + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(com=period - 1, adjust=False).mean()
        return adx

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns for multiple windows, candle body/wick ratios, daily range position."""
        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']

        for p in [1, 3, 5, 10, 20]:
            df[f'return_{p}'] = close.pct_change(p)

        candle_range = (high - low).replace(0, np.nan)
        body = (close - open_).abs()
        df['body_ratio'] = body / candle_range
        df['upper_wick'] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / candle_range
        df['lower_wick'] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / candle_range
        df['candle_direction'] = np.sign(close - open_)

        daily_high = high.rolling(96).max()   # ~24h of 15m bars
        daily_low = low.rolling(96).min()
        daily_range = (daily_high - daily_low).replace(0, np.nan)
        df['price_position'] = (close - daily_low) / daily_range

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume ratio, OBV, volume momentum."""
        volume = df['volume']
        close = df['close']

        vol_ma20 = volume.rolling(20).mean().replace(0, np.nan)
        df['volume_ratio'] = volume / vol_ma20

        direction = np.sign(close.diff())
        df['obv'] = (volume * direction).cumsum()
        df['obv_ma20'] = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'] / (df['obv_ma20'].replace(0, np.nan))

        df['volume_momentum'] = volume.pct_change(5)
        df['volume_std'] = volume.rolling(20).std() / vol_ma20

        return df

    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Up/down bar counts, distance from EMAs, volatility regime."""
        close = df['close']
        direction = (close.diff() > 0).astype(int)

        df['up_bars_10'] = direction.rolling(10).sum()
        df['down_bars_10'] = 10 - df['up_bars_10']
        df['up_bars_20'] = direction.rolling(20).sum()

        df['dist_ema50'] = (close - df['ema_50']) / df['ema_50']
        df['dist_ema200'] = (close - df['ema_200']) / df['ema_200']
        df['ema50_above_200'] = (df['ema_50'] > df['ema_200']).astype(int)

        returns = close.pct_change()
        vol_short = returns.rolling(10).std()
        vol_long = returns.rolling(50).std()
        df['volatility_regime'] = vol_short / (vol_long.replace(0, np.nan))
        df['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(96)  # annualise to daily

        return df

    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features: cyclic encoding + session binary flags."""
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                df['hour'] = 0
                df['hour_sin'] = 0.0
                df['hour_cos'] = 1.0
                df['dow_sin'] = 0.0
                df['dow_cos'] = 1.0
                for flag in ['is_london', 'is_ny', 'is_asia', 'is_overlap']:
                    df[flag] = 0
                return df

        hour = df.index.hour
        dow = df.index.dayofweek

        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        df['is_asia'] = ((hour >= 0) & (hour < 8)).astype(int)    # UTC 00-08
        df['is_london'] = ((hour >= 8) & (hour < 16)).astype(int)  # UTC 08-16
        df['is_ny'] = ((hour >= 13) & (hour < 21)).astype(int)     # UTC 13-21
        df['is_overlap'] = ((hour >= 13) & (hour < 16)).astype(int)  # London/NY overlap UTC

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_numeric_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return sorted list of numeric feature columns (exclude OHLCV originals)."""
        exclude = {'open', 'high', 'low', 'close', 'volume', 'timestamp', 'label'}
        cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        return cols
