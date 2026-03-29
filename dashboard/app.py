"""Flask web dashboard for monitoring the trading bot."""

import os
import json
from datetime import datetime, timezone

from flask import Flask, render_template, jsonify

from portfolio_tracker import PortfolioTracker

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Bootstrap portfolio tracker from saved state
# ---------------------------------------------------------------------------

_state_file = os.getenv('PORTFOLIO_STATE', './data/portfolio_state.json')
_initial_balance = float(os.getenv('INITIAL_BALANCE', '10000'))
_portfolio = PortfolioTracker(initial_balance=_initial_balance, state_file=_state_file)


def _get_portfolio() -> PortfolioTracker:
    """Return the global portfolio tracker (reloads state each request)."""
    _portfolio.load_state()
    return _portfolio


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Main dashboard page."""
    try:
        p = _get_portfolio()
        context = {
            'balance': p.current_balance,
            'total_pnl': p.get_total_pnl(),
            'win_rate': p.get_win_rate(),
            'drawdown': p.get_max_drawdown(),
            'sharpe': p.get_sharpe_ratio(),
            'open_positions': len(p.open_positions),
            'total_trades': len(p.closed_trades),
            'mode': os.getenv('TRADING_MODE', 'paper').upper(),
        }
    except Exception as e:
        context = {'error': str(e)}
    return render_template('index.html', **context)


@app.route('/positions')
def positions():
    """Open positions page."""
    try:
        p = _get_portfolio()
        positions_list = list(p.open_positions.values())
    except Exception:
        positions_list = []
    return render_template('positions.html', positions=positions_list)


@app.route('/trades')
def trades():
    """Trade history page."""
    try:
        p = _get_portfolio()
        trades_list = list(reversed(p.closed_trades))
        total_pnl = p.get_total_pnl()
        win_rate = p.get_win_rate()
        pf = p.get_profit_factor()
        profit_factor = None if pf == float('inf') or pf != pf else round(pf, 2)
    except Exception:
        trades_list, total_pnl, win_rate, profit_factor = [], 0.0, 0.0, None
    return render_template(
        'trades.html',
        trades=trades_list,
        total_pnl=total_pnl,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


@app.route('/models')
def models():
    """Model performance page."""
    metrics = _load_model_metrics()
    return render_template('models.html', metrics=metrics)


@app.route('/assets')
def assets():
    """Asset performance breakdown page."""
    try:
        p = _get_portfolio()
        asset_perf = p.get_asset_performance()
    except Exception:
        asset_perf = {}
    return render_template('assets.html', asset_perf=asset_perf)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route('/api/status')
def api_status():
    """JSON status endpoint."""
    try:
        p = _get_portfolio()
        eq = p.get_equity_curve()
        equity_data = []
        if not eq.empty:
            equity_data = [
                {'t': str(row['timestamp']), 'v': row['balance']}
                for _, row in eq.iterrows()
            ]
        return jsonify({
            'status': 'running',
            'mode': os.getenv('TRADING_MODE', 'paper'),
            'balance': p.current_balance,
            'total_pnl': p.get_total_pnl(),
            'win_rate': p.get_win_rate(),
            'drawdown': p.get_max_drawdown(),
            'sharpe': p.get_sharpe_ratio(),
            'open_positions': len(p.open_positions),
            'total_trades': len(p.closed_trades),
            'equity_curve': equity_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/models/performance')
def api_models_performance():
    """JSON model performance endpoint."""
    metrics = _load_model_metrics()
    return jsonify(metrics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_metrics() -> dict:
    """Load model training reports from disk."""
    logs_dir = os.getenv('LOGS_PATH', './logs')
    metrics = {}
    for symbol in ['BTCUSDT', 'XAUUSDT']:
        path = os.path.join(logs_dir, f'training_report_{symbol}.json')
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                metrics[symbol] = data.get('metrics', {})
            except Exception:
                pass
    return metrics


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
