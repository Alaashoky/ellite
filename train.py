"""
train.py — Standalone training script for the Ellite trading bot.

This script ties together the full AI training pipeline with strategy signals:
  - Fetches historical OHLCV data (ccxt/MT5)
  - Runs FeatureEngineer to produce base features
  - Optionally enriches features with MarketStructureDetector and
    ElliottWaveDetector signals
  - Trains XGBoost, RandomForest, and LSTM models via ModelTrainer
  - Optionally runs walk-forward validation, strategy signal analysis,
    and/or a quick backtest

Usage examples:
    python train.py --symbol BTCUSDT --start-date 2021-01-01 --end-date 2024-01-01 --strategy-analysis
    python train.py --symbol XAUUSDT --data-source mt5 --mt5-symbol XAUUSD --walk-forward --n-splits 5
    # Train on 2020-2024, backtest automatically on 2024→today (out-of-sample):
    python train.py --symbol BTCUSDT --start-date 2020-01-01 --end-date 2024-01-01 --backtest
    # Or specify a custom backtest end date:
    python train.py --symbol BTCUSDT --start-date 2020-01-01 --end-date 2024-01-01 --backtest --backtest-end-date 2025-06-01
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Logging setup (console + file).  File handler is added after args are parsed
# so we know the symbol and logs-dir.
# ---------------------------------------------------------------------------

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("train")


def _setup_logging(logs_dir: str, symbol: str) -> None:
    """Configure root logger to write to console and a timestamped file."""
    os.makedirs(logs_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"train_{symbol}_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FMT,
        datefmt=_DATE_FMT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger.info("Logging initialized — file: %s", log_file)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AI models for the Ellite trading bot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        choices=["BTCUSDT", "XAUUSDT"],
        help="Trading symbol to train on.",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Start date for historical data (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default="2024-01-01",
        help="End date for historical data (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--timeframe",
        default="15m",
        help="Candle timeframe (e.g. 1m, 5m, 15m, 1h, 4h).",
    )
    parser.add_argument(
        "--data-source",
        default="ccxt",
        choices=["ccxt", "mt5"],
        help="Data source to use.",
    )
    parser.add_argument(
        "--mt5-symbol",
        default=None,
        help="MT5 symbol override (e.g. XAUUSD.m). Required when --data-source=mt5.",
    )
    parser.add_argument(
        "--models-dir",
        default="./saved_models",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--logs-dir",
        default="./logs",
        help="Directory for log files and training reports.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation after training.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of splits for walk-forward validation.",
    )
    parser.add_argument(
        "--strategy-analysis",
        action="store_true",
        help="Run strategy signal analysis (market structure + Elliott waves) on the fetched data.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run a quick backtest after training using the backtester.",
    )
    parser.add_argument(
        "--backtest-end-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help=(
            "End date for the out-of-sample backtest (YYYY-MM-DD). "
            "Defaults to today's date. The backtest start date is "
            "always set to --end-date so there is no overlap with training data."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Strategy analysis helpers
# ---------------------------------------------------------------------------

def _run_strategy_analysis(df, symbol: str):
    """
    Run MarketStructureDetector + ElliottWaveDetector on *df* and print a
    summary report.  Returns a dict with the extracted features/stats so the
    caller can log them, or None if the trading_strategy imports are unavailable.
    """
    try:
        from trading_strategy.market_structure import MarketStructureDetector
        from trading_strategy.elliott_wave import ElliottWaveDetector
    except ImportError as exc:
        logger.warning("trading_strategy imports failed — skipping strategy analysis. (%s)", exc)
        return None

    logger.info("=== Strategy Signal Analysis ===")

    # ---- Market Structure ----
    ms_detector = MarketStructureDetector(data=df)
    swing_df = ms_detector.detect_swing_points(df)
    structures = ms_detector.detect_market_structure(swing_df)
    bias = ms_detector.get_current_bias(structures)

    bos_count = sum(1 for s in structures if getattr(s, "structure_type", "") == "BOS")
    choch_count = sum(1 for s in structures if getattr(s, "structure_type", "") == "CHoCH")

    # ---- Elliott Wave ----
    ew_detector = ElliottWaveDetector(data=df)
    waves = ew_detector.find_elliott_wave_sequence(0, len(df))
    ew_wave_count = len(waves)
    ew_last_wave = getattr(waves[-1], "wave_number", None) if waves else None

    # ---- Summary ----
    logger.info("--- Market Structure Summary ---")
    logger.info("  BOS structures detected : %d", bos_count)
    logger.info("  CHoCH structures detected: %d", choch_count)
    logger.info("  Current HTF bias        : %s", bias)
    logger.info("--- Elliott Wave Summary ---")
    logger.info("  Waves found  : %d", ew_wave_count)
    logger.info("  Last wave num: %s", ew_last_wave)

    print("\n=== Strategy Signal Analysis Report ===")
    print(f"  Symbol              : {symbol}")
    print(f"  BOS detected        : {bos_count}")
    print(f"  CHoCH detected      : {choch_count}")
    print(f"  Current HTF bias    : {bias}")
    print(f"  Elliott waves found : {ew_wave_count}")
    print(f"  Last wave number    : {ew_last_wave}")
    print("=" * 42)

    return {
        "bos_count": bos_count,
        "choch_count": choch_count,
        "htf_bias": bias,
        "ew_wave_count": ew_wave_count,
        "ew_last_wave": ew_last_wave,
    }


def _build_strategy_features(df, symbol: str):
    """
    Enrich *df* with strategy-derived columns:
      - htf_bias      : int  (BULLISH=1, BEARISH=-1, NEUTRAL=0)
      - ew_wave_count : int  (number of Elliott waves found)
      - ew_last_wave  : int/None (last detected wave number)

    Returns the enriched DataFrame.  If imports fail, returns *df* unchanged
    with a warning.
    """
    import pandas as pd

    try:
        from trading_strategy.market_structure import MarketStructureDetector
        from trading_strategy.elliott_wave import ElliottWaveDetector
    except ImportError as exc:
        logger.warning(
            "trading_strategy imports failed — strategy features will not be added. (%s)", exc
        )
        return df

    logger.info("Building strategy-enriched features for %s …", symbol)

    # Market Structure → htf_bias
    ms_detector = MarketStructureDetector(data=df)
    swing_df = ms_detector.detect_swing_points(df)
    structures = ms_detector.detect_market_structure(swing_df)
    bias_str = ms_detector.get_current_bias(structures)

    bias_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
    df = df.copy()
    df["htf_bias"] = bias_map.get(bias_str, 0)

    # Elliott Wave → ew_wave_count, ew_last_wave
    ew_detector = ElliottWaveDetector(data=df)
    waves = ew_detector.find_elliott_wave_sequence(0, len(df))
    df["ew_wave_count"] = len(waves)
    df["ew_last_wave"] = (
        getattr(waves[-1], "wave_number", 0) if waves else 0
    )

    logger.info(
        "Strategy features added — bias=%s (%d), ew_wave_count=%d, ew_last_wave=%s",
        bias_str,
        df["htf_bias"].iloc[0],
        df["ew_wave_count"].iloc[0],
        df["ew_last_wave"].iloc[0],
    )
    return df


# ---------------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------------

def _run_quick_backtest(symbol: str, backtest_start_date: str, backtest_end_date: str):
    """Run an out-of-sample quick backtest using UnifiedBacktester and print the result.

    The backtest intentionally uses a date range that does **not** overlap with
    the training period.  ``backtest_start_date`` should equal the training
    ``end_date`` so the model is evaluated on data it has never seen.

    Args:
        symbol: Trading pair symbol.
        backtest_start_date: Start of the backtest window (= end of training).
        backtest_end_date: End of the backtest window (defaults to today).
    """
    try:
        from backtest import UnifiedBacktester
    except ImportError as exc:
        logger.warning("backtest module unavailable — skipping backtest. (%s)", exc)
        return

    logger.info(
        "Backtest  range : %s → %s  (out-of-sample)",
        backtest_start_date,
        backtest_end_date,
    )
    logger.info("=== Quick Backtest ===")
    bt = UnifiedBacktester(base_path=".")
    result = bt.quick_validation(
        pair=symbol, start_date=backtest_start_date, end_date=backtest_end_date
    )

    if result:
        print("\n=== Quick Backtest Results ===")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print("=" * 32)
    else:
        logger.warning("Backtest returned no results.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # Logging must be set up before anything else logs.
    _setup_logging(args.logs_dir, args.symbol)

    logger.info("=== Ellite Training Script ===")
    logger.info("Symbol       : %s", args.symbol)
    logger.info("Date range   : %s → %s", args.start_date, args.end_date)
    logger.info("Timeframe    : %s", args.timeframe)
    logger.info("Data source  : %s", args.data_source)
    if args.mt5_symbol:
        logger.info("MT5 symbol   : %s", args.mt5_symbol)
    logger.info("Models dir   : %s", args.models_dir)
    logger.info("Logs dir     : %s", args.logs_dir)

    try:
        from ai_models.model_trainer import ModelTrainer

        trainer = ModelTrainer(
            symbol=args.symbol,
            models_dir=args.models_dir,
            logs_dir=args.logs_dir,
        )

        # ------------------------------------------------------------------
        # 1. Fetch historical data
        # ------------------------------------------------------------------
        logger.info("Fetching historical data …")
        df = trainer.fetch_historical_data(
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            data_source=args.data_source,
            mt5_symbol=args.mt5_symbol,
        )
        logger.info("Data fetched: %d rows", len(df))

        # ------------------------------------------------------------------
        # 2. Full training pipeline
        #    ModelTrainer.train_all_models() performs its own feature
        #    engineering internally, ensuring consistent feature generation
        #    between training and inference.  We pass the raw OHLCV df so
        #    the trainer controls the full pipeline end-to-end.
        # ------------------------------------------------------------------
        logger.info("Starting model training …")
        metrics = trainer.train_all_models(df=df)

        # Print training metrics
        print("\n=== Training Results ===")
        for model_name, model_metrics in metrics.items():
            if "error" in model_metrics:
                print(f"  {model_name:15s}: ERROR — {model_metrics['error']}")
                logger.warning("%s training error: %s", model_name, model_metrics["error"])
            else:
                acc = model_metrics.get("accuracy", "N/A")
                print(f"  {model_name:15s}: accuracy = {acc}")
                logger.info("%s accuracy: %s", model_name, acc)
        print("=" * 25)

        # ------------------------------------------------------------------
        # 4. Walk-forward validation
        # ------------------------------------------------------------------
        if args.walk_forward:
            logger.info("Running walk-forward validation (n_splits=%d) …", args.n_splits)
            wf_results = trainer.run_walk_forward_validation(df, n_splits=args.n_splits)

            print("\n=== Walk-Forward Validation ===")
            print(f"  Mean accuracy : {wf_results['mean_accuracy']:.4f}")
            for i, fold in enumerate(wf_results.get("fold_metrics", []), start=1):
                fold_acc = fold.get("accuracy", "N/A")
                print(f"  Fold {i:2d}       : accuracy = {fold_acc}")
            print("=" * 32)

            logger.info(
                "Walk-forward mean accuracy: %.4f over %d folds",
                wf_results["mean_accuracy"],
                len(wf_results.get("fold_metrics", [])),
            )

        # ------------------------------------------------------------------
        # 5. Strategy signal analysis
        # ------------------------------------------------------------------
        if args.strategy_analysis:
            # Build strategy-enriched features only when needed to avoid
            # unnecessary computation when --strategy-analysis is not set.
            df_enriched = _build_strategy_features(df, args.symbol)
            _run_strategy_analysis(df_enriched, args.symbol)

        # ------------------------------------------------------------------
        # 6. Quick backtest (out-of-sample: end_date → backtest_end_date)
        # ------------------------------------------------------------------
        if args.backtest:
            logger.info(
                "Training  range : %s → %s",
                args.start_date,
                args.end_date,
            )
            _run_quick_backtest(args.symbol, args.end_date, args.backtest_end_date)

        logger.info("Training pipeline completed successfully.")

    except Exception as exc:
        logger.exception("Fatal error during training pipeline: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
