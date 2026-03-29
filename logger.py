"""Loguru-based logging with multiple sinks for the trading bot."""

import json
import sys
import os
from typing import Optional

from loguru import logger as _logger

# Expose the configured logger as the module-level logger
logger = _logger


def setup_logger(
    logs_dir: str = './logs',
    log_level: str = 'DEBUG',
) -> None:
    """
    Configure loguru with four sinks:

    1. Coloured console output at INFO level.
    2. Rotating file (trading.log) — max 50 MB, retained 30 days.
    3. errors.log — ERROR and above only.
    4. trades.log — trade records as JSON lines.
    """
    os.makedirs(logs_dir, exist_ok=True)

    _logger.remove()  # remove default handler

    # 1. Console — coloured, human readable
    _logger.add(
        sys.stdout,
        level='INFO',
        colorize=True,
        format=(
            '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
            '<level>{level: <8}</level> | '
            '<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>'
        ),
    )

    # 2. Rolling general log
    _logger.add(
        os.path.join(logs_dir, 'trading.log'),
        level=log_level,
        rotation='50 MB',
        retention='30 days',
        compression='gz',
        enqueue=True,
        format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}',
    )

    # 3. Errors only
    _logger.add(
        os.path.join(logs_dir, 'errors.log'),
        level='ERROR',
        rotation='10 MB',
        retention='60 days',
        compression='gz',
        enqueue=True,
        format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}\n{exception}',
    )

    # 4. Trade records — JSON lines (filter by bound extra key 'trade_record')
    _logger.add(
        os.path.join(logs_dir, 'trades.log'),
        level='INFO',
        filter=lambda record: record['extra'].get('trade_record', False),
        format='{message}',
        rotation='50 MB',
        retention='90 days',
        enqueue=True,
        serialize=False,
    )

    _logger.info(f"Logger initialised. Logs directory: {os.path.abspath(logs_dir)}")


def log_trade(trade_data: dict) -> None:
    """
    Append a trade record as a JSON line to trades.log.

    Args:
        trade_data: dict containing trade fields (symbol, side, entry, exit, pnl, …)
    """
    _logger.bind(trade_record=True).info(json.dumps(trade_data, default=str))


def log_performance(metrics: dict) -> None:
    """
    Log a performance snapshot at INFO level.

    Args:
        metrics: dict with performance metrics
    """
    _logger.info(f"[PERFORMANCE] {json.dumps(metrics, default=str)}")
