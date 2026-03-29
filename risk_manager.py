"""Risk management: position sizing, drawdown control, daily risk limits."""

import logging
from datetime import date
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforces risk rules before any trade is placed.

    Tracks daily PnL, drawdown, and open position count.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        risk_per_trade: float = 0.01,
        max_daily_risk: float = 0.05,
        max_drawdown: float = 0.15,
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_drawdown = max_drawdown

        self.daily_pnl: float = 0.0
        self.daily_reset_date: date = date.today()
        self.open_positions: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self, entry_price: float, stop_loss: float, symbol: str = 'BTCUSDT'
    ) -> float:
        """
        Compute position size based on fixed-fractional risk.

        size = (balance * risk_per_trade) / |entry - stop_loss|

        Returns:
            Quantity of the base asset to buy/sell.
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
        risk_amount = self.current_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        if price_risk < 1e-8:
            return 0.0

        from config.settings import ASSETS_CONFIG
        asset_cfg = ASSETS_CONFIG.get(symbol, {})
        min_order = asset_cfg.get('min_order_size', 0.001)
        qty_precision = asset_cfg.get('quantity_precision', 5)

        raw_qty = risk_amount / price_risk
        qty = round(raw_qty, qty_precision)
        return max(qty, min_order)

    # ------------------------------------------------------------------
    # Risk limit checks
    # ------------------------------------------------------------------

    def check_daily_risk_limit(self) -> bool:
        """
        Return True if daily losses are within the allowed limit.

        Resets the daily counter at the start of a new calendar day.
        """
        self._auto_reset_daily()
        daily_loss_pct = -self.daily_pnl / max(self.current_balance, 1e-8)
        return daily_loss_pct < self.max_daily_risk

    def check_max_drawdown(self) -> bool:
        """Return True if current drawdown is below the maximum allowed."""
        return self.get_current_drawdown() < self.max_drawdown

    def check_max_concurrent_positions(self, max_positions: int = 3) -> bool:
        """Return True if the number of open positions is below the cap."""
        return len(self.open_positions) < max_positions

    def validate_trade(self, signal: str, confidence: float) -> Tuple[bool, str]:
        """
        Validate all risk conditions before entering a trade.

        Returns:
            (allowed, reason_message)
        """
        from config.settings import RISK_CONFIG
        min_conf = RISK_CONFIG.get('min_confidence', 0.65)

        if confidence < min_conf:
            return False, f"Confidence {confidence:.2%} below minimum {min_conf:.2%}"
        if not self.check_daily_risk_limit():
            return False, f"Daily risk limit reached ({self.max_daily_risk:.0%})"
        if not self.check_max_drawdown():
            return False, f"Max drawdown breached ({self.get_current_drawdown():.2%})"
        if not self.check_max_concurrent_positions():
            return False, "Max concurrent positions reached"
        return True, "OK"

    # ------------------------------------------------------------------
    # Balance & position tracking
    # ------------------------------------------------------------------

    def update_balance(self, pnl: float) -> None:
        """Record a realised PnL and update tracking variables."""
        self.current_balance += pnl
        self.daily_pnl += pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

    def get_current_drawdown(self) -> float:
        """Return current drawdown as a fraction of peak balance."""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance

    def add_position(self, symbol: str, position_data: dict) -> None:
        """Register an open position."""
        self.open_positions[symbol] = position_data

    def remove_position(self, symbol: str) -> Optional[dict]:
        """Deregister a closed position, returning its data."""
        return self.open_positions.pop(symbol, None)

    def reset_daily_pnl(self) -> None:
        """Manually reset the daily PnL counter."""
        self.daily_pnl = 0.0
        self.daily_reset_date = date.today()

    def get_risk_summary(self) -> Dict:
        """Return a snapshot of current risk state."""
        return {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': self.get_current_drawdown(),
            'daily_pnl': self.daily_pnl,
            'open_positions': len(self.open_positions),
            'daily_risk_ok': self.check_daily_risk_limit(),
            'drawdown_ok': self.check_max_drawdown(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auto_reset_daily(self) -> None:
        today = date.today()
        if today != self.daily_reset_date:
            logger.info("Daily PnL counter reset.")
            self.reset_daily_pnl()
