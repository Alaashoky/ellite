"""Portfolio tracker with full metrics, persistence, and HTML/CSV reporting."""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class PortfolioTracker:
    """
    Tracks open positions, closed trades, equity curve, and risk metrics.
    Persists state to JSON for crash recovery.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        state_file: str = './data/portfolio_state.json',
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.state_file = state_file
        self._makedirs_for(state_file)

        self.open_positions: Dict[str, dict] = {}
        self.closed_trades: List[dict] = []
        self._equity_curve: List[dict] = [
            {'timestamp': datetime.now(timezone.utc).isoformat(), 'balance': initial_balance}
        ]
        self.load_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _makedirs_for(path: str) -> None:
        """Create parent directories for a file path."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        amount: float,
        stop_loss: float,
        take_profits: list,
        model_signal: str = '',
        confidence: float = 0.0,
    ) -> dict:
        """Register a new open position."""
        position = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'amount': amount,
            'stop_loss': stop_loss,
            'take_profits': list(take_profits),
            'current_price': entry_price,
            'unrealised_pnl': 0.0,
            'model_signal': model_signal,
            'confidence': confidence,
            'opened_at': datetime.now(timezone.utc).isoformat(),
        }
        self.open_positions[symbol] = position
        self.save_state()
        return position

    def close_position(
        self, symbol: str, exit_price: float, reason: str = 'manual'
    ) -> Optional[dict]:
        """Close a position, record the trade, and update balance."""
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return None
        trade = self._record_trade(pos, exit_price, pos['amount'], reason)
        self._update_equity()
        self.save_state()
        return trade

    def partial_close(
        self, symbol: str, exit_price: float, close_fraction: float = 0.5
    ) -> Optional[dict]:
        """Close a fraction of an open position."""
        pos = self.open_positions.get(symbol)
        if pos is None:
            return None
        close_amount = pos['amount'] * close_fraction
        trade = self._record_trade(pos, exit_price, close_amount, 'partial_tp')
        pos['amount'] -= close_amount
        if pos['amount'] <= 0:
            del self.open_positions[symbol]
        self._update_equity()
        self.save_state()
        return trade

    def update_prices(self, prices: dict) -> None:
        """Update unrealised PnL for all open positions."""
        for symbol, price in prices.items():
            pos = self.open_positions.get(symbol)
            if pos is None:
                continue
            pos['current_price'] = price
            if pos['side'] == 'buy':
                pos['unrealised_pnl'] = (price - pos['entry_price']) * pos['amount']
            else:
                pos['unrealised_pnl'] = (pos['entry_price'] - price) * pos['amount']

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_total_pnl(self) -> float:
        """Sum of all realised PnL from closed trades."""
        return sum(t['pnl'] for t in self.closed_trades)

    def get_win_rate(self) -> float:
        """Fraction of winning trades."""
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t['pnl'] > 0)
        return wins / len(self.closed_trades)

    def get_profit_factor(self) -> float:
        """Gross profits divided by gross losses."""
        gross_profit = sum(t['pnl'] for t in self.closed_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.closed_trades if t['pnl'] < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualised Sharpe ratio from per-trade returns."""
        returns = self._trade_returns()
        if len(returns) < 2:
            return 0.0
        mean = np.mean(returns) - risk_free_rate
        std = np.std(returns, ddof=1)
        return float(mean / std * np.sqrt(252)) if std > 0 else 0.0

    def get_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualised Sortino ratio (downside deviation)."""
        returns = self._trade_returns()
        if len(returns) < 2:
            return 0.0
        mean = np.mean(returns) - risk_free_rate
        neg_returns = [r for r in returns if r < 0]
        downside = np.std(neg_returns, ddof=1) if neg_returns else 1e-8
        return float(mean / downside * np.sqrt(252))

    def get_max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown of the equity curve."""
        eq = self.get_equity_curve()
        if eq.empty:
            return 0.0
        balances = eq['balance'].values
        peak = np.maximum.accumulate(balances)
        drawdowns = (peak - balances) / np.where(peak > 0, peak, 1)
        return float(drawdowns.max())

    def get_calmar_ratio(self) -> float:
        """Annualised return divided by max drawdown."""
        mdd = self.get_max_drawdown()
        if mdd == 0:
            return 0.0
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        return float(total_return / mdd)

    def get_asset_performance(self) -> Dict:
        """Per-symbol breakdown: trade count, win rate, total PnL."""
        symbols = {t['symbol'] for t in self.closed_trades}
        result = {}
        for sym in symbols:
            trades = [t for t in self.closed_trades if t['symbol'] == sym]
            pnl = sum(t['pnl'] for t in trades)
            wins = sum(1 for t in trades if t['pnl'] > 0)
            result[sym] = {
                'total_trades': len(trades),
                'win_rate': wins / len(trades) if trades else 0.0,
                'total_pnl': pnl,
            }
        return result

    def get_daily_report(self) -> Dict:
        """Summary statistics for today's session."""
        today = datetime.now(timezone.utc).date()
        today_trades = [
            t for t in self.closed_trades
            if datetime.fromisoformat(t['closed_at']).date() == today
        ]
        daily_pnl = sum(t['pnl'] for t in today_trades)
        wins = sum(1 for t in today_trades if t['pnl'] > 0)
        return {
            'date': today.isoformat(),
            'balance': self.current_balance,
            'daily_pnl': daily_pnl,
            'total_trades': len(today_trades),
            'win_rate': wins / len(today_trades) if today_trades else 0.0,
            'drawdown': self.get_max_drawdown(),
            'open_positions': len(self.open_positions),
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as a DataFrame with timestamp and balance columns."""
        df = pd.DataFrame(self._equity_curve)
        if df.empty:
            return df
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_html_report(self, output_path: str = './reports/report.html') -> str:
        """Generate a Plotly-based HTML report with equity curve and drawdown."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return '<html><body><p>plotly not installed.</p></body></html>'

        self._makedirs_for(output_path)
        eq = self.get_equity_curve()
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Equity Curve', 'Drawdown (%)'),
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.65, 0.35],
        )

        if not eq.empty:
            fig.add_trace(
                go.Scatter(x=eq['timestamp'], y=eq['balance'],
                           mode='lines', name='Balance', line=dict(color='royalblue')),
                row=1, col=1,
            )
            peak = eq['balance'].cummax()
            drawdown_pct = (peak - eq['balance']) / peak.replace(0, 1) * 100
            fig.add_trace(
                go.Scatter(x=eq['timestamp'], y=-drawdown_pct,
                           mode='lines', fill='tozeroy', name='Drawdown',
                           line=dict(color='tomato')),
                row=2, col=1,
            )

        stats = (
            f"Total PnL: {self.get_total_pnl():.2f} | "
            f"Win Rate: {self.get_win_rate():.1%} | "
            f"Sharpe: {self.get_sharpe_ratio():.2f} | "
            f"Max DD: {self.get_max_drawdown():.2%}"
        )
        fig.update_layout(
            title=f'Portfolio Report — {stats}',
            height=700,
            showlegend=True,
        )

        html = fig.to_html(full_html=True, include_plotlyjs='cdn')
        with open(output_path, 'w') as f:
            f.write(html)
        return output_path

    def export_to_csv(self, output_path: str = './data/trades.csv') -> str:
        """Export all closed trades to CSV."""
        self._makedirs_for(output_path)
        df = pd.DataFrame(self.closed_trades)
        df.to_csv(output_path, index=False)
        return output_path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist portfolio state to JSON."""
        state = {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'open_positions': self.open_positions,
            'closed_trades': self.closed_trades,
            'equity_curve': self._equity_curve,
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self) -> None:
        """Load portfolio state from JSON if the file exists."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            self.current_balance = state.get('current_balance', self.initial_balance)
            self.open_positions = state.get('open_positions', {})
            self.closed_trades = state.get('closed_trades', [])
            self._equity_curve = state.get('equity_curve', self._equity_curve)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load portfolio state: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_trade(
        self, pos: dict, exit_price: float, amount: float, reason: str
    ) -> dict:
        """Build a closed trade record and update current_balance."""
        if pos['side'] == 'buy':
            pnl = (exit_price - pos['entry_price']) * amount
        else:
            pnl = (pos['entry_price'] - exit_price) * amount

        trade = {
            'id': pos.get('id', str(uuid.uuid4())),
            'symbol': pos['symbol'],
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'amount': amount,
            'pnl': pnl,
            'pnl_pct': pnl / (pos['entry_price'] * amount) if pos['entry_price'] * amount else 0.0,
            'model_signal': pos.get('model_signal', ''),
            'confidence': pos.get('confidence', 0.0),
            'reason': reason,
            'opened_at': pos.get('opened_at', ''),
            'closed_at': datetime.now(timezone.utc).isoformat(),
        }
        self.closed_trades.append(trade)
        self.current_balance += pnl
        return trade

    def _update_equity(self) -> None:
        self._equity_curve.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'balance': self.current_balance,
        })

    def _trade_returns(self) -> List[float]:
        """Return list of per-trade PnL-% values."""
        return [t.get('pnl_pct', 0.0) for t in self.closed_trades]
