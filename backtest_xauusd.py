#!/usr/bin/env python3
"""
backtest_xauusd.py — Out-of-sample backtest script for XAUUSD (Gold).

Runs a full out-of-sample backtest on XAUUSD.m (Gold) for the period
2024-01-01 → 2026-03-30 using the existing UnifiedBacktester from backtest.py.

The model was trained on 2020-01-01 → 2024-01-01, so data after 2024-01-01
is the true out-of-sample period.

Usage examples:
    python backtest_xauusd.py
    python backtest_xauusd.py --mode quick
    python backtest_xauusd.py --mode multi
    python backtest_xauusd.py --mode walkforward
    python backtest_xauusd.py --start-date 2024-01-01 --end-date 2025-01-01 --initial-balance 50000
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Logging setup — same pattern as train.py
# ---------------------------------------------------------------------------

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("backtest_xauusd")


def _setup_logging(logs_dir: str) -> None:
    """Configure root logger to write to console and a timestamped file."""
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"backtest_xauusd_{ts}.log")

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
        description="Out-of-sample backtest for XAUUSD (Gold) using UnifiedBacktester.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        default="XAUUSD",
        help="Pair name used in model/config.",
    )
    parser.add_argument(
        "--mt5-symbol",
        default="XAUUSD.m",
        help="Actual MT5 symbol used for data loading.",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Out-of-sample backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default="2026-03-30",
        help="Out-of-sample backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000,
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
        help="Directory containing saved models.",
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

def _save_report(result, report_dir: str, mode: str) -> str:
    """Save backtest result to a timestamped JSON file. Returns the file path."""
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"xauusd_{mode}_{ts}.json"
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
    # Extract metrics — handle both dict and other return types
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
    print("╔══════════════════════════════════════╗")
    print("║   XAUUSD Out-of-Sample Backtest      ║")
    print(f"║   Period : {args.start_date} → {args.end_date}   ║")
    print(f"║   Mode   : {args.mode:<26}║")
    print("╚══════════════════════════════════════╝")
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
    result = bt.quick_validation(
        pair=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return result


def _run_single(bt, args) -> dict:
    logger.info("Running SINGLE period backtest for %s", args.symbol)
    result = bt.single_period_backtest(
        pair=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        detailed=True,
    )
    return result


def _run_multi(bt, args) -> dict:
    """Run multi-period backtest over Gold-specific quarterly periods."""
    periods = {
        "Q1_2024": ("2024-01-01", "2024-03-31"),
        "Q2_2024": ("2024-04-01", "2024-06-30"),
        "Q3_2024": ("2024-07-01", "2024-09-30"),
        "Q4_2024": ("2024-10-01", "2024-12-31"),
        "Q1_2025": ("2025-01-01", "2025-03-31"),
        "Q2_2025": ("2025-04-01", "2025-06-30"),
        "Q3_2025": ("2025-07-01", "2025-09-30"),
        "Q4_2025": ("2025-10-01", "2025-12-31"),
        "Q1_2026": ("2026-01-01", "2026-03-30"),
    }
    logger.info("Running MULTI period backtest for %s (%d periods)", args.symbol, len(periods))
    result = bt.multi_period_backtest(pair=args.symbol, periods=periods)
    return result


def _run_walkforward(bt, args) -> dict:
    logger.info("Running WALK-FORWARD analysis for %s", args.symbol)
    result = bt.walk_forward_analysis(
        pair=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        train_months=3,
        test_months=1,
    )
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _setup_logging(args.logs_dir)

    logger.info("=== XAUUSD Out-of-Sample Backtest ===")
    logger.info("Symbol      : %s (MT5: %s)", args.symbol, args.mt5_symbol)
    logger.info("Period      : %s → %s", args.start_date, args.end_date)
    logger.info("Mode        : %s", args.mode)
    logger.info("Balance     : $%.2f", args.initial_balance)
    logger.info("Models dir  : %s", args.models_dir)
    logger.info("Report dir  : %s", args.report_dir)

    # Import UnifiedBacktester — fail clearly if unavailable
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

        # Save report
        report_path = _save_report(result, args.report_dir, args.mode)
        logger.info("Backtest complete. Report: %s", report_path)

        # Print summary
        _print_summary(args, result)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error during backtest: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
