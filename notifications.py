"""Non-blocking Telegram notifications via daemon threads."""

import json
import logging
import threading
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = 'https://api.telegram.org/bot{token}/sendMessage'


class TelegramNotifier:
    """
    Send Telegram messages asynchronously without blocking the trading loop.

    All public send_* methods dispatch a daemon thread so the caller returns
    immediately even when Telegram is slow or unreachable.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        if not self._enabled:
            logger.info('TelegramNotifier: token/chat_id not configured — notifications disabled.')

    # ------------------------------------------------------------------
    # Public notification methods
    # ------------------------------------------------------------------

    def send_trade_alert(self, trade_data: dict) -> None:
        """Send a trade entry/exit alert."""
        symbol = trade_data.get('symbol', '?')
        side = trade_data.get('side', '?').upper()
        price = trade_data.get('price', trade_data.get('entry_price', 0))
        pnl = trade_data.get('pnl')
        conf = trade_data.get('confidence', 0)

        lines = [
            f"🤖 *Trade Alert* — {symbol}",
            f"Side      : {side}",
            f"Price     : {price:.4f}",
            f"Confidence: {conf:.1%}",
        ]
        if pnl is not None:
            emoji = '✅' if pnl >= 0 else '❌'
            lines.append(f"PnL       : {emoji} {pnl:+.2f} USDT")
        self._dispatch(self._format_md('\n'.join(lines)))

    def send_daily_report(self, report: dict) -> None:
        """Send end-of-day performance summary."""
        balance = report.get('balance', 0)
        daily_pnl = report.get('daily_pnl', 0)
        win_rate = report.get('win_rate', 0)
        trades = report.get('total_trades', 0)
        drawdown = report.get('drawdown', 0)

        emoji = '📈' if daily_pnl >= 0 else '📉'
        text = (
            f"{emoji} *Daily Report*\n"
            f"Balance  : {balance:.2f} USDT\n"
            f"Daily PnL: {daily_pnl:+.2f} USDT\n"
            f"Win Rate : {win_rate:.1%}\n"
            f"Trades   : {trades}\n"
            f"Drawdown : {drawdown:.2%}"
        )
        self._dispatch(self._format_md(text))

    def send_error_alert(self, error: str) -> None:
        """Send an error / exception alert."""
        text = f"🚨 *Error Alert*\n```\n{error[:2000]}\n```"
        self._dispatch(text)

    def send_model_performance(self, metrics: dict) -> None:
        """Send model accuracy / performance update."""
        lines = ['📊 *Model Performance*']
        for model_name, m in metrics.items():
            acc = m.get('accuracy', 0)
            lines.append(f"  {model_name}: {acc:.2%} accuracy")
        self._dispatch(self._format_md('\n'.join(lines)))

    def send_startup(self, assets: List[str], mode: str) -> None:
        """Send bot startup notification."""
        asset_str = ', '.join(assets)
        text = (
            f"🚀 *Bot Started*\n"
            f"Mode  : {mode.upper()}\n"
            f"Assets: {asset_str}"
        )
        self._dispatch(self._format_md(text))

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _send_message(self, text: str) -> None:
        """Perform the actual HTTP POST to the Telegram Bot API."""
        if not self._enabled:
            return
        url = _TELEGRAM_API.format(token=self.bot_token)
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if not resp.ok:
                logger.warning(f"Telegram API error {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Telegram send failed: {e}")

    def _dispatch(self, text: str) -> None:
        """Launch a daemon thread to send a message without blocking."""
        t = threading.Thread(target=self._send_message, args=(text,), daemon=True)
        t.start()

    @staticmethod
    def _format_md(text: str) -> str:
        """Return text as-is (Telegram Markdown v1 is permissive for bot messages)."""
        return text
