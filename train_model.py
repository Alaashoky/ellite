"""
train_model.py — Dedicated in-sample model training script for the Ellite trading bot.

Default training window: 2020-01-01 → 2023-12-31
The resulting models are saved to ``saved_models/`` and are intended to be
evaluated on the out-of-sample period (2024-01-01 → …) via ``backtest_oos.py``.

Usage examples:
    # Train with defaults (2020-01-01 → 2023-12-31, BTCUSDT, 15m):
    python train_model.py

    # Train a specific symbol on a custom window:
    python train_model.py --symbol XAUUSDT --start-date 2019-01-01 --end-date 2023-12-31

    # Train using MT5 data:
    python train_model.py --symbol XAUUSDT --data-source mt5 --mt5-symbol XAUUSD.m

    # Train with walk-forward validation:
    python train_model.py --symbol BTCUSDT --walk-forward --n-splits 5
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

logger = logging.getLogger("train_model")


def _setup_logging(logs_dir: str, symbol: str) -> None:
    """Configure root logger to write to console and a timestamped file."""
    os.makedirs(logs_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"train_model_{symbol}_{ts}.log")

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
        description=(
            "Train AI models on in-sample data for the Ellite trading bot. "
            "Default window: 2020-01-01 → 2023-12-31."
        ),
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
        help="Start date for historical training data (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default="2023-12-31",
        help="End date for historical training data (YYYY-MM-DD).",
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
        help="Directory to save trained model artifacts.",
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

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------

def _validate_dates(start_date: str, end_date: str) -> None:
    """Parse and validate that start_date <= end_date; exit with a clear error otherwise."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid --start-date format '%s'. Use YYYY-MM-DD.", start_date)
        sys.exit(1)

    try:
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid --end-date format '%s'. Use YYYY-MM-DD.", end_date)
        sys.exit(1)

    if start > end:
        logger.error(
            "--start-date (%s) must be on or before --end-date (%s).",
            start_date,
            end_date,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Logging must be set up before anything else logs.
    _setup_logging(args.logs_dir, args.symbol)

    logger.info("=== Ellite Model Training (In-Sample) ===")
    logger.info("Symbol       : %s", args.symbol)
    logger.info("Date range   : %s → %s", args.start_date, args.end_date)
    logger.info("Timeframe    : %s", args.timeframe)
    logger.info("Data source  : %s", args.data_source)
    if args.mt5_symbol:
        logger.info("MT5 symbol   : %s", args.mt5_symbol)
    logger.info("Models dir   : %s", args.models_dir)
    logger.info("Logs dir     : %s", args.logs_dir)

    # Validate date range before any expensive operations.
    _validate_dates(args.start_date, args.end_date)

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
        # 2. Train all models
        # ------------------------------------------------------------------
        logger.info("Starting model training …")
        metrics = trainer.train_all_models(df=df)

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
        # 3. Walk-forward validation (optional)
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

        logger.info(
            "Training pipeline completed successfully. "
            "Models saved to: %s",
            args.models_dir,
        )
        logger.info(
            "Next step: run 'python backtest_oos.py --symbol %s' to evaluate "
            "on out-of-sample data (2024-01-01 → today).",
            args.symbol,
        )

    except Exception as exc:
        logger.exception("Fatal error during training pipeline: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
