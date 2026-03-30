#!/usr/bin/env python3
"""
backtest_oos.py — Dedicated out-of-sample backtest script for the Ellite trading bot.

Default backtest window: 2024-01-01 → today (dynamically resolved at runtime).
This window intentionally starts where the default training window ends (2023-12-31)
so there is no overlap between training and evaluation data.

Usage examples:
    # Run backtest with defaults (2024-01-01 → today, BTCUSDT, single mode):
    python backtest_oos.py

    # Backtest XAUUSD on a specific period:
    python backtest_oos.py --symbol XAUUSD --start-date 2024-01-01 --end-date 2025-01-01

    # Quick validation mode with a custom initial balance:
    python backtest_oos.py --mode quick --initial-balance 50000

    # Walk-forward analysis:
    python backtest_oos.py --symbol BTCUSDT --mode walkforward

    # Save reports to a custom directory:
    python backtest_oos.py --report-dir ./my_reports
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Logging setup — same pattern as train_model.py
# ---------------------------------------------------------------------------

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("backtest_oos")


def _setup_logging(logs_dir: str, symbol: str) -> None:
    """Configure root logger to write to console and a timestamped file."""
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"backtest_oos_{symbol}_{ts}.log")

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
    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description=(
            "Out-of-sample backtest for the Ellite trading bot. "
            "Default window: 2024-01-01 → today."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair symbol (e.g. BTCUSDT, XAUUSD).",
    )
    parser.add_argument(
        "--mt5-symbol",
        default=None,
        help="Actual MT5 symbol if different from --symbol (e.g. XAUUSD.m).",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Out-of-sample backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=_today,
        help="Out-of-sample backtest end date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Initial account balance in USD.",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "single", "multi", "walkforward"],
        default="single",
        help="Backtest mode to run.",
    )
    parser.add_argument(
        "--models-dir",
        default="./saved_models",
        help="Directory containing saved model artifacts.",
    )
    parser.add_argument(
        "--logs-dir",
        default="./logs",
        help="Directory for log files.",
    )
    parser.add_argument(
        "--report-dir",
        default="./backtest_reports",
        help="Directory to save backtest report JSON files.",
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
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _make_serializable(obj):
    """Recursively convert non-JSON-serializable types to native Python types."""
    try:
        import pandas as pd
        import numpy as np

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------

def _save_report(result, report_dir: str, symbol: str, mode: str) -> str:
    """Save backtest result to a timestamped JSON file. Returns the file path."""
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_oos_{symbol}_{mode}_{ts}.json"
    filepath = os.path.join(report_dir, filename)

    serializable_result = _make_serializable(result)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(serializable_result, fh, indent=2, default=str)

    logger.info("Report saved → %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(args: argparse.Namespace, result) -> None:
    """Print a formatted summary box with key backtest metrics."""
    if isinstance(result, dict):
        final_balance = result.get("final_balance", args.initial_balance)
        total_trades = result.get("total_trades", 0)
        win_rate = result.get("win_rate") or result.get("win_rate_pct", 0.0)
        profit_factor = result.get("profit_factor", 0.0)
        max_drawdown = result.get("max_drawdown") or result.get("max_drawdown_pct", 0.0)
        sharpe = result.get("sharpe_ratio", 0.0)
        return_pct = result.get("return_pct") or result.get("total_return_pct", 0.0)
        if not return_pct and final_balance != args.initial_balance:
            return_pct = (final_balance - args.initial_balance) / args.initial_balance * 100
    else:
        final_balance = args.initial_balance
        total_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        max_drawdown = 0.0
        sharpe = 0.0
        return_pct = 0.0

    sign = "+" if return_pct >= 0 else ""

    print()
    print("╔══════════════════════════════════════════╗")
    print("║   Out-of-Sample Backtest Results         ║")
    print(f"║   Symbol : {args.symbol:<30}║")
    print(f"║   Period : {args.start_date} → {args.end_date}     ║")
    print(f"║   Mode   : {args.mode:<30}║")
    print("╚══════════════════════════════════════════╝")
    print()
    print(f"  Initial Balance : ${args.initial_balance:,.2f}")
    print(f"  Final Balance   : ${final_balance:,.2f}")
    print(f"  Total Return    : {sign}{return_pct:.2f}%")
    print(f"  Total Trades    : {total_trades}")
    print(f"  Win Rate        : {win_rate:.2f}%")
    print(f"  Profit Factor   : {profit_factor:.2f}")
    print(f"  Max Drawdown    : {max_drawdown:.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.2f}")
    print()


# ---------------------------------------------------------------------------
# Backtest runners
# ---------------------------------------------------------------------------

def _run_quick(bt, args) -> dict:
    logger.info("Running QUICK validation backtest for %s", args.symbol)
    return bt.quick_validation(
        pair=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )


def _run_single(bt, args) -> dict:
    logger.info("Running SINGLE period backtest for %s", args.symbol)
    return bt.single_period_backtest(
        pair=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        detailed=True,
    )


def _run_multi(bt, args) -> dict:
    """Build quarterly periods from start_date → end_date and run multi-period backtest."""
    import pandas as pd

    quarters = pd.period_range(start=args.start_date, end=args.end_date, freq="Q")
    periods = {}
    for q in quarters:
        key = str(q)
        q_start = q.start_time.strftime("%Y-%m-%d")
        q_end = min(q.end_time, pd.Timestamp(args.end_date)).strftime("%Y-%m-%d")
        periods[key] = (q_start, q_end)

    if not periods:
        logger.warning(
            "No complete quarters found in range %s → %s; falling back to SINGLE mode.",
            args.start_date,
            args.end_date,
        )
        return _run_single(bt, args)

    logger.info(
        "Running MULTI period backtest for %s (%d quarters)",
        args.symbol,
        len(periods),
    )
    return bt.multi_period_backtest(pair=args.symbol, periods=periods)


def _run_walkforward(bt, args) -> dict:
    logger.info("Running WALK-FORWARD analysis for %s", args.symbol)
    return bt.walk_forward_analysis(
        pair=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        train_months=3,
        test_months=1,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _setup_logging(args.logs_dir, args.symbol)

    logger.info("=== Ellite Out-of-Sample Backtest ===")
    logger.info("Symbol      : %s", args.symbol)
    if args.mt5_symbol:
        logger.info("MT5 symbol  : %s", args.mt5_symbol)
    logger.info("Period      : %s → %s", args.start_date, args.end_date)
    logger.info("Mode        : %s", args.mode)
    logger.info("Balance     : $%.2f", args.initial_balance)
    logger.info("Models dir  : %s", args.models_dir)
    logger.info("Report dir  : %s", args.report_dir)

    # Validate date range before any expensive operations.
    _validate_dates(args.start_date, args.end_date)

    # Import UnifiedBacktester — fail clearly if unavailable.
    try:
        from backtest import UnifiedBacktester
    except ImportError as exc:
        logger.error(
            "Cannot import UnifiedBacktester from backtest.py: %s\n"
            "Make sure backtest.py is in the same directory or on PYTHONPATH.",
            exc,
        )
        sys.exit(1)

    try:
        bt = UnifiedBacktester(
            base_path=".",
            initial_balance=args.initial_balance,
        )

        mode_runners = {
            "quick": _run_quick,
            "single": _run_single,
            "multi": _run_multi,
            "walkforward": _run_walkforward,
        }

        result = mode_runners[args.mode](bt, args)

        # Save JSON report with timestamp and key metrics.
        report_path = _save_report(result, args.report_dir, args.symbol, args.mode)
        logger.info("Backtest complete. Report: %s", report_path)

        # Print human-readable summary.
        _print_summary(args, result)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error during backtest: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
