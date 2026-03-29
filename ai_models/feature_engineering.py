"""Feature engineering for AI trading models."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    from trading_strategy.market_structure import MarketStructureDetector
    from trading_strategy.elliott_wave import ElliottWaveDetector
    from trading_strategy.ict_concepts import ICTConceptsDetector
    _STRATEGY_AVAILABLE = True
except Exception:
    _STRATEGY_AVAILABLE = False


class FeatureEngineer:
    """Generates technical and statistical features for ML models."""

    def __init__(self, symbol: str):
        """
        Initialize FeatureEngineer for a specific trading symbol.

        Args:
            symbol: Trading symbol, e.g. 'BTCUSDT' or 'XAUUSDT'
        """
        self.symbol = symbol
        self.is_btc = 'BTC' in symbol.upper()
        self.is_gold = 'XAU' in symbol.upper() or 'GOLD' in symbol.upper()
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
        # NEW: Add strategy-based features
        if _STRATEGY_AVAILABLE:
            df = self._add_ict_features(df)
            df = self._add_elliott_wave_features(df)
            df = self._add_advanced_market_structure_features(df)
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

    def _add_ict_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ICT (Inner Circle Trader) concept features:
        - Order Blocks (bullish/bearish)
        - Fair Value Gaps (bullish/bearish)
        - Liquidity levels (above/below)
        - Kill zones (Asia/London/NY sessions)
        """
        try:
            detector = ICTConceptsDetector()

            # --- Order Blocks ---
            # Bullish OB: bearish candle before a strong bullish move
            # Bearish OB: bullish candle before a strong bearish move
            bullish_ob = pd.Series(0, index=df.index)
            bearish_ob = pd.Series(0, index=df.index)
            ob_strength = pd.Series(0.0, index=df.index)

            close = df['close']
            open_ = df['open']
            high = df['high']
            low = df['low']

            for i in range(2, len(df) - 1):
                # Bullish OB: bearish candle followed by strong bullish move
                if (open_.iloc[i] > close.iloc[i] and  # bearish candle
                    close.iloc[i+1] > high.iloc[i] and  # next breaks above
                    (close.iloc[i+1] - open_.iloc[i+1]) > (high.iloc[i] - low.iloc[i]) * 0.5):  # strong move
                    bullish_ob.iloc[i] = 1
                    ob_strength.iloc[i] = min((close.iloc[i+1] - open_.iloc[i+1]) / (high.iloc[i] - low.iloc[i] + 1e-10), 3.0)

                # Bearish OB: bullish candle followed by strong bearish move
                if (close.iloc[i] > open_.iloc[i] and  # bullish candle
                    close.iloc[i+1] < low.iloc[i] and  # next breaks below
                    (open_.iloc[i+1] - close.iloc[i+1]) > (high.iloc[i] - low.iloc[i]) * 0.5):  # strong move
                    bearish_ob.iloc[i] = 1
                    ob_strength.iloc[i] = min((open_.iloc[i+1] - close.iloc[i+1]) / (high.iloc[i] - low.iloc[i] + 1e-10), 3.0)

            df['ict_bullish_ob'] = bullish_ob
            df['ict_bearish_ob'] = bearish_ob
            df['ict_ob_strength'] = ob_strength

            # Active OB zone (price is inside an OB zone - rolling window of last 20 candles)
            df['ict_in_bullish_ob_zone'] = bullish_ob.rolling(20).max()
            df['ict_in_bearish_ob_zone'] = bearish_ob.rolling(20).max()

            # --- Fair Value Gaps (FVG) ---
            # Bullish FVG: gap between candle[i-1].high and candle[i+1].low (3-candle pattern)
            # Bearish FVG: gap between candle[i-1].low and candle[i+1].high
            bullish_fvg = pd.Series(0, index=df.index)
            bearish_fvg = pd.Series(0, index=df.index)
            fvg_size = pd.Series(0.0, index=df.index)

            for i in range(1, len(df) - 1):
                # Bullish FVG
                gap_bull = low.iloc[i+1] - high.iloc[i-1]
                if gap_bull > 0:
                    bullish_fvg.iloc[i] = 1
                    fvg_size.iloc[i] = gap_bull / (close.iloc[i] + 1e-10)

                # Bearish FVG
                gap_bear = low.iloc[i-1] - high.iloc[i+1]
                if gap_bear > 0:
                    bearish_fvg.iloc[i] = 1
                    fvg_size.iloc[i] = -gap_bear / (close.iloc[i] + 1e-10)

            df['ict_bullish_fvg'] = bullish_fvg
            df['ict_bearish_fvg'] = bearish_fvg
            df['ict_fvg_size'] = fvg_size

            # Unfilled FVG nearby (rolling 30 candles)
            df['ict_unfilled_bullish_fvg'] = bullish_fvg.rolling(30).max()
            df['ict_unfilled_bearish_fvg'] = bearish_fvg.rolling(30).max()

            # --- Liquidity Levels ---
            # Equal highs (buy-side liquidity) and equal lows (sell-side liquidity)
            tolerance = close * 0.001  # 0.1% tolerance

            equal_highs = pd.Series(0, index=df.index)
            equal_lows = pd.Series(0, index=df.index)

            for i in range(5, len(df)):
                # Equal highs in last 20 candles (buy-side liquidity above)
                recent_highs = high.iloc[max(0, i-20):i]
                if (abs(high.iloc[i] - recent_highs.max()) < tolerance.iloc[i] and
                        recent_highs.max() == recent_highs.iloc[-1]):
                    equal_highs.iloc[i] = 1

                # Equal lows in last 20 candles (sell-side liquidity below)
                recent_lows = low.iloc[max(0, i-20):i]
                if (abs(low.iloc[i] - recent_lows.min()) < tolerance.iloc[i] and
                        recent_lows.min() == recent_lows.iloc[-1]):
                    equal_lows.iloc[i] = 1

            df['ict_equal_highs'] = equal_highs  # Buy-side liquidity above
            df['ict_equal_lows'] = equal_lows    # Sell-side liquidity below

            # Liquidity sweep detection
            # Price sweeps equal highs then reverses
            liquidity_sweep_high = pd.Series(0, index=df.index)
            liquidity_sweep_low = pd.Series(0, index=df.index)

            for i in range(3, len(df)):
                if equal_highs.iloc[i-1] == 1 and close.iloc[i] < high.iloc[i-1]:
                    liquidity_sweep_high.iloc[i] = 1  # Swept highs, bearish reversal
                if equal_lows.iloc[i-1] == 1 and close.iloc[i] > low.iloc[i-1]:
                    liquidity_sweep_low.iloc[i] = 1   # Swept lows, bullish reversal

            df['ict_liquidity_sweep_high'] = liquidity_sweep_high
            df['ict_liquidity_sweep_low'] = liquidity_sweep_low

            # --- Premium/Discount zones ---
            # Based on 50% level of recent swing range (last 50 candles)
            swing_high_50 = high.rolling(50).max()
            swing_low_50 = low.rolling(50).min()
            swing_mid_50 = (swing_high_50 + swing_low_50) / 2
            df['ict_in_premium'] = (close > swing_mid_50).astype(int)   # Above 50% = premium (sell zone)
            df['ict_in_discount'] = (close < swing_mid_50).astype(int)  # Below 50% = discount (buy zone)
            df['ict_pd_ratio'] = (close - swing_low_50) / (swing_high_50 - swing_low_50 + 1e-10)  # 0=low, 1=high

        except Exception:
            # If anything fails, add zeros
            for col in ['ict_bullish_ob', 'ict_bearish_ob', 'ict_ob_strength',
                        'ict_in_bullish_ob_zone', 'ict_in_bearish_ob_zone',
                        'ict_bullish_fvg', 'ict_bearish_fvg', 'ict_fvg_size',
                        'ict_unfilled_bullish_fvg', 'ict_unfilled_bearish_fvg',
                        'ict_equal_highs', 'ict_equal_lows',
                        'ict_liquidity_sweep_high', 'ict_liquidity_sweep_low',
                        'ict_in_premium', 'ict_in_discount', 'ict_pd_ratio']:
                df[col] = 0
        return df

    def _add_elliott_wave_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Elliott Wave features:
        - Current wave number (1-5, A-C)
        - Wave direction (impulse vs corrective)
        - Fibonacci levels proximity
        - Wave completion probability
        """
        try:
            # Use a rolling window to detect Elliott Wave patterns
            close = df['close']
            high = df['high']
            low = df['low']

            # Wave identification using swing points (simplified for ML features)
            # Use multiple lookback periods to identify wave structure
            wave_number = pd.Series(0, index=df.index)
            wave_bullish = pd.Series(0, index=df.index)
            wave_impulse = pd.Series(0, index=df.index)
            fib_0382_proximity = pd.Series(0.0, index=df.index)
            fib_0618_proximity = pd.Series(0.0, index=df.index)
            fib_1618_proximity = pd.Series(0.0, index=df.index)
            wave_momentum_div = pd.Series(0, index=df.index)

            lookback = 50  # candles to analyze for wave structure

            for i in range(lookback, len(df)):
                segment = df.iloc[i-lookback:i+1]
                seg_close = segment['close']
                seg_high = segment['high']
                seg_low = segment['low']

                # Find significant swing points in segment
                swings = []
                for j in range(2, len(segment) - 2):
                    h = seg_high.iloc[j]
                    l = seg_low.iloc[j]
                    # Swing high
                    if h > seg_high.iloc[j-1] and h > seg_high.iloc[j-2] and h > seg_high.iloc[j+1] and h > seg_high.iloc[j+2]:
                        swings.append(('H', h, j))
                    # Swing low
                    elif l < seg_low.iloc[j-1] and l < seg_low.iloc[j-2] and l < seg_low.iloc[j+1] and l < seg_low.iloc[j+2]:
                        swings.append(('L', l, j))

                if len(swings) < 3:
                    continue

                # Determine wave count from last 5 swings
                recent_swings = swings[-5:] if len(swings) >= 5 else swings
                current_price = seg_close.iloc[-1]
                first_swing_price = recent_swings[0][1]
                last_swing_price = recent_swings[-1][1]

                # Determine overall direction
                is_bullish = last_swing_price > first_swing_price
                wave_bullish.iloc[i] = int(is_bullish)

                # Count wave number based on number of alternating swings
                n_swings = len(recent_swings)
                if n_swings >= 5:
                    # Check for 5-wave impulse pattern
                    types = [s[0] for s in recent_swings[-5:]]
                    if types in [['L','H','L','H','L'], ['H','L','H','L','H']]:
                        wave_number.iloc[i] = 5
                        wave_impulse.iloc[i] = 1
                    elif n_swings >= 3:
                        wave_number.iloc[i] = 3
                        wave_impulse.iloc[i] = 0  # likely corrective ABC
                elif n_swings == 3:
                    wave_number.iloc[i] = 3
                elif n_swings == 2:
                    wave_number.iloc[i] = 2
                elif n_swings == 1:
                    wave_number.iloc[i] = 1

                # Fibonacci proximity
                if len(recent_swings) >= 2:
                    swing_start = recent_swings[-2][1]
                    swing_end = recent_swings[-1][1]
                    price_range = abs(swing_end - swing_start)

                    if price_range > 0:
                        # Distance from key Fibonacci levels (normalized)
                        fib_382 = swing_start + (swing_end - swing_start) * 0.382 if is_bullish else swing_start - (swing_start - swing_end) * 0.382
                        fib_618 = swing_start + (swing_end - swing_start) * 0.618 if is_bullish else swing_start - (swing_start - swing_end) * 0.618
                        fib_1618_ext = swing_end + price_range * 0.618

                        fib_0382_proximity.iloc[i] = 1 - min(abs(current_price - fib_382) / price_range, 1.0)
                        fib_0618_proximity.iloc[i] = 1 - min(abs(current_price - fib_618) / price_range, 1.0)
                        fib_1618_proximity.iloc[i] = 1 - min(abs(current_price - fib_1618_ext) / price_range, 1.0)

                # Momentum divergence (price makes new extreme but RSI doesn't)
                if i >= 14:
                    rsi_segment = close.iloc[i-14:i+1]
                    delta = rsi_segment.diff()
                    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
                    rs = gain / loss.replace(0, np.nan)
                    rsi_vals = 100 - (100 / (1 + rs))
                    if len(rsi_vals) >= 2:
                        price_trend = close.iloc[i] - close.iloc[i-5]
                        rsi_trend = rsi_vals.iloc[-1] - rsi_vals.iloc[-6] if len(rsi_vals) >= 6 else 0
                        wave_momentum_div.iloc[i] = int((price_trend > 0 and rsi_trend < 0) or
                                                        (price_trend < 0 and rsi_trend > 0))

            df['ew_wave_number'] = wave_number
            df['ew_wave_bullish'] = wave_bullish
            df['ew_wave_impulse'] = wave_impulse
            df['ew_fib_0382_proximity'] = fib_0382_proximity
            df['ew_fib_0618_proximity'] = fib_0618_proximity
            df['ew_fib_1618_proximity'] = fib_1618_proximity
            df['ew_momentum_divergence'] = wave_momentum_div

            # Wave 3 signal (strongest wave — best entry)
            df['ew_is_wave3'] = (wave_number == 3).astype(int)
            # Wave 5 with divergence (reversal signal)
            df['ew_wave5_reversal'] = ((wave_number == 5) & (wave_momentum_div == 1)).astype(int)
            # ABC correction end (potential reversal/continuation entry)
            df['ew_abc_complete'] = ((wave_number == 3) & (wave_impulse == 0)).astype(int)

        except Exception:
            for col in ['ew_wave_number', 'ew_wave_bullish', 'ew_wave_impulse',
                        'ew_fib_0382_proximity', 'ew_fib_0618_proximity', 'ew_fib_1618_proximity',
                        'ew_momentum_divergence', 'ew_is_wave3', 'ew_wave5_reversal', 'ew_abc_complete']:
                df[col] = 0
        return df

    def _add_advanced_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced market structure features using MarketStructureDetector:
        - Break of Structure (BOS)
        - Change of Character (CHoCH)
        - Higher Highs / Lower Lows sequence
        - Market bias (bullish/bearish/neutral)
        - Swing point proximity
        """
        try:
            close = df['close']
            high = df['high']
            low = df['low']

            # Detect swing points (simple implementation for feature engineering)
            strength = 3  # candles on each side

            swing_high = pd.Series(False, index=df.index)
            swing_low = pd.Series(False, index=df.index)
            swing_high_price = pd.Series(np.nan, index=df.index)
            swing_low_price = pd.Series(np.nan, index=df.index)

            h_vals = high.values
            l_vals = low.values

            for i in range(strength, len(df) - strength):
                is_sh = all(h_vals[i] > h_vals[i-j-1] and h_vals[i] > h_vals[i+j+1] for j in range(strength))
                is_sl = all(l_vals[i] < l_vals[i-j-1] and l_vals[i] < l_vals[i+j+1] for j in range(strength))
                if is_sh:
                    swing_high.iloc[i] = True
                    swing_high_price.iloc[i] = h_vals[i]
                if is_sl:
                    swing_low.iloc[i] = True
                    swing_low_price.iloc[i] = l_vals[i]

            # BOS detection: price breaks previous swing high/low
            bos_bullish = pd.Series(0, index=df.index)  # Break above previous swing high
            bos_bearish = pd.Series(0, index=df.index)  # Break below previous swing low
            choch = pd.Series(0, index=df.index)         # Change of character

            # Track last swing high and low
            last_sh = np.nan
            last_sl = np.nan
            prev_last_sh = np.nan
            prev_last_sl = np.nan
            trend = 0  # 1=bullish, -1=bearish

            for i in range(len(df)):
                if swing_high.iloc[i]:
                    prev_last_sh = last_sh
                    last_sh = swing_high_price.iloc[i]

                if swing_low.iloc[i]:
                    prev_last_sl = last_sl
                    last_sl = swing_low_price.iloc[i]

                if not np.isnan(last_sh) and close.iloc[i] > last_sh:
                    if trend == -1:
                        choch.iloc[i] = 1   # CHoCH: was bearish, now bullish break
                    else:
                        bos_bullish.iloc[i] = 1  # BOS: continuation bullish
                    trend = 1

                if not np.isnan(last_sl) and close.iloc[i] < last_sl:
                    if trend == 1:
                        choch.iloc[i] = 1   # CHoCH: was bullish, now bearish break
                    else:
                        bos_bearish.iloc[i] = 1  # BOS: continuation bearish
                    trend = -1

            df['ms_bos_bullish'] = bos_bullish
            df['ms_bos_bearish'] = bos_bearish
            df['ms_choch'] = choch

            # Higher High / Lower Low sequence (rolling 20 candles)
            hh = pd.Series(0, index=df.index)
            hl = pd.Series(0, index=df.index)
            lh = pd.Series(0, index=df.index)
            ll = pd.Series(0, index=df.index)

            sh_prices = swing_high_price.dropna()
            sl_prices = swing_low_price.dropna()

            for i in range(1, len(sh_prices)):
                idx = sh_prices.index[i]
                if sh_prices.iloc[i] > sh_prices.iloc[i-1]:
                    hh[idx] = 1
                else:
                    lh[idx] = 1

            for i in range(1, len(sl_prices)):
                idx = sl_prices.index[i]
                if sl_prices.iloc[i] > sl_prices.iloc[i-1]:
                    hl[idx] = 1
                else:
                    ll[idx] = 1

            df['ms_higher_high'] = hh
            df['ms_higher_low'] = hl
            df['ms_lower_high'] = lh
            df['ms_lower_low'] = ll

            # Market bias score (rolling 10 structures)
            bullish_score = (bos_bullish + choch * 0.5).rolling(10).sum()
            bearish_score = (bos_bearish + choch * 0.5).rolling(10).sum()
            df['ms_bias_score'] = (bullish_score - bearish_score) / (bullish_score + bearish_score + 1e-10)

            # Proximity to nearest swing high/low (normalized by ATR)
            nearest_sh = swing_high_price.ffill()
            nearest_sl = swing_low_price.ffill()
            atr = df.get('atr_14', (high - low).rolling(14).mean())

            df['ms_dist_to_swing_high'] = (nearest_sh - close) / (atr + 1e-10)
            df['ms_dist_to_swing_low'] = (close - nearest_sl) / (atr + 1e-10)

            # Structure trend (rolling count of HH+HL vs LH+LL)
            bull_struct = (hh + hl).rolling(20).sum()
            bear_struct = (lh + ll).rolling(20).sum()
            df['ms_structure_trend'] = (bull_struct - bear_struct) / (bull_struct + bear_struct + 1e-10)

        except Exception:
            for col in ['ms_bos_bullish', 'ms_bos_bearish', 'ms_choch',
                        'ms_higher_high', 'ms_higher_low', 'ms_lower_high', 'ms_lower_low',
                        'ms_bias_score', 'ms_dist_to_swing_high', 'ms_dist_to_swing_low',
                        'ms_structure_trend']:
                df[col] = 0
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_numeric_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return sorted list of numeric feature columns (exclude OHLCV originals)."""
        exclude = {'open', 'high', 'low', 'close', 'volume', 'timestamp', 'label'}
        cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        return cols
