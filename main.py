"""CLI entry point for the Ellite trading bot."""

import argparse
import os
import sys


def _load_env() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='ellite',
        description='Ellite AI Trading Bot — BTC/USDT & XAU/USDT',
    )
    subparsers = parser.add_subparsers(dest='command')

    # ------------------------------------------------------------------
    # Legacy positional-mode interface (train / paper / live / backtest / dashboard)
    # ------------------------------------------------------------------
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['train', 'paper', 'live', 'backtest', 'dashboard'],
        help='Operating mode (legacy positional argument)',
    )
    parser.add_argument(
        '--assets',
        default=os.getenv('ASSETS', 'BTCUSDT,XAUUSDT'),
        help='Comma-separated list of assets (default: BTCUSDT,XAUUSDT)',
    )
    parser.add_argument(
        '--balance',
        type=float,
        default=float(os.getenv('INITIAL_BALANCE', '10000')),
        help='Initial paper-trade balance (default: 10000)',
    )
    parser.add_argument(
        '--model',
        default=os.getenv('AI_MODEL', 'ensemble'),
        choices=['ensemble', 'lstm', 'xgboost', 'random_forest'],
        help='AI model to use (default: ensemble)',
    )
    parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='Training start date YYYY-MM-DD (train/backtest modes)',
    )
    parser.add_argument(
        '--end-date',
        default='2024-01-01',
        help='Training end date YYYY-MM-DD (train/backtest modes)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Dashboard port (default: 5000)',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity',
    )

    # ------------------------------------------------------------------
    # train-mt5 sub-command
    # ------------------------------------------------------------------
    p_train_mt5 = subparsers.add_parser(
        'train-mt5',
        help='Train AI models using MetaTrader 5 historical data.',
    )
    p_train_mt5.add_argument(
        '--symbols',
        default='EURUSD,XAUUSD',
        help='Comma-separated MT5 symbol names (default: EURUSD,XAUUSD)',
    )
    p_train_mt5.add_argument(
        '--start',
        default='2020-01-01',
        help='Training start date YYYY-MM-DD (default: 2020-01-01)',
    )
    p_train_mt5.add_argument(
        '--end',
        default='2023-12-31',
        help='Training end date YYYY-MM-DD (default: 2023-12-31)',
    )
    p_train_mt5.add_argument(
        '--timeframe',
        default='15m',
        help='Candlestick timeframe (default: 15m)',
    )
    p_train_mt5.add_argument(
        '--model',
        default=os.getenv('AI_MODEL', 'ensemble'),
        choices=['ensemble', 'lstm', 'xgboost', 'random_forest'],
        help='AI model to use (default: ensemble)',
    )
    p_train_mt5.add_argument(
        '--balance',
        type=float,
        default=float(os.getenv('INITIAL_BALANCE', '10000')),
        help='Initial balance for sizing context (default: 10000)',
    )
    p_train_mt5.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity',
    )

    # ------------------------------------------------------------------
    # live-mt5 sub-command
    # ------------------------------------------------------------------
    p_live_mt5 = subparsers.add_parser(
        'live-mt5',
        help='Start the MT5-backed live/paper trading loop.',
    )
    p_live_mt5.add_argument(
        '--symbols',
        default='EURUSD,XAUUSD',
        help='Comma-separated MT5 symbol names (default: EURUSD,XAUUSD)',
    )
    p_live_mt5.add_argument(
        '--mode',
        default='paper',
        choices=['paper', 'live'],
        help='Trading mode: paper (default) or live',
    )
    p_live_mt5.add_argument(
        '--balance',
        type=float,
        default=float(os.getenv('INITIAL_BALANCE', '10000')),
        help='Initial balance (default: 10000)',
    )
    p_live_mt5.add_argument(
        '--model',
        default=os.getenv('AI_MODEL', 'ensemble'),
        choices=['ensemble', 'lstm', 'xgboost', 'random_forest'],
        help='AI model to use (default: ensemble)',
    )
    p_live_mt5.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity',
    )

    return parser.parse_args()


def _run_train(args) -> None:
    from logger import setup_logger
    setup_logger(log_level=args.log_level)
    from ai_models.model_trainer import ModelTrainer
    assets = [a.strip() for a in args.assets.split(',')]
    for symbol in assets:
        print(f"\n{'='*60}")
        print(f"Training models for {symbol}")
        print('='*60)
        trainer = ModelTrainer(symbol)
        metrics = trainer.train_all_models(
            start_date=args.start_date,
            end_date=args.end_date,
        )
        for model_name, m in metrics.items():
            acc = m.get('accuracy', 'N/A')
            print(f"  {model_name}: accuracy={acc}")
    print("\nTraining complete.")


def _run_paper(args) -> None:
    from logger import setup_logger
    setup_logger(log_level=args.log_level)
    from live_trader import LiveTrader
    assets = [a.strip() for a in args.assets.split(',')]
    trader = LiveTrader(
        assets=assets,
        model_type=args.model,
        mode='paper',
        initial_balance=args.balance,
    )
    try:
        trader.run()
    except KeyboardInterrupt:
        trader.stop()
        print("\nPaper trading stopped.")


def _run_live(args) -> None:
    from logger import setup_logger
    setup_logger(log_level=args.log_level)
    api_key = os.getenv('BINANCE_API_KEY', '')
    secret_key = os.getenv('BINANCE_SECRET_KEY', '')
    if not api_key or not secret_key:
        print("ERROR: BINANCE_API_KEY and BINANCE_SECRET_KEY must be set for live mode.")
        sys.exit(1)
    from live_trader import LiveTrader
    assets = [a.strip() for a in args.assets.split(',')]
    trader = LiveTrader(
        assets=assets,
        model_type=args.model,
        mode='live',
        initial_balance=args.balance,
    )
    try:
        trader.run()
    except KeyboardInterrupt:
        trader.stop()
        print("\nLive trading stopped.")


def _run_backtest(args) -> None:
    from logger import setup_logger
    setup_logger(log_level=args.log_level)
    try:
        from backtester import Backtester
    except ImportError:
        print("ERROR: backtester module not found. Run 'train' first.")
        sys.exit(1)
    assets = [a.strip() for a in args.assets.split(',')]
    for symbol in assets:
        print(f"\nBacktesting {symbol} from {args.start_date} to {args.end_date}")
        bt = Backtester(symbol=symbol, initial_balance=args.balance)
        results = bt.run(start_date=args.start_date, end_date=args.end_date)
        print(f"  PnL: {results.get('total_pnl', 0):.2f}")
        print(f"  Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"  Sharpe: {results.get('sharpe', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")


def _run_train_mt5(args) -> None:
    """Train AI models using MetaTrader 5 historical data."""
    from logger import setup_logger
    setup_logger(log_level=args.log_level)
    from ai_models.model_trainer import ModelTrainer
    symbols = [s.strip() for s in args.symbols.split(',')]
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Training models for {symbol} (MT5)")
        print('='*60)
        trainer = ModelTrainer(symbol)
        metrics = trainer.train_all_models(
            start_date=args.start,
            end_date=args.end,
            data_source='mt5',
            mt5_symbol=symbol,
        )
        for model_name, m in metrics.items():
            acc = m.get('accuracy', 'N/A')
            print(f"  {model_name}: accuracy={acc}")
    print("\nMT5 training complete.")


def _run_live_mt5(args) -> None:
    """Start the MT5-backed live/paper trading loop."""
    from logger import setup_logger
    setup_logger(log_level=args.log_level)
    from mt5_live_trader import MT5LiveTrader
    symbols = [s.strip() for s in args.symbols.split(',')]
    trader = MT5LiveTrader(
        assets=symbols,
        model_type=args.model,
        mode=args.mode,
        initial_balance=args.balance,
    )
    try:
        trader.run()
    except KeyboardInterrupt:
        trader.stop()
        print("\nMT5 trading stopped.")



    from logger import setup_logger
    setup_logger(log_level=args.log_level)
    print(f"Starting dashboard on http://0.0.0.0:{args.port}")
    try:
        from dashboard.app import app
        app.run(host='0.0.0.0', port=args.port, debug=False)
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main() -> None:
    _load_env()
    args = _parse_args()

    # Handle new sub-commands first
    if args.command == 'train-mt5':
        try:
            _run_train_mt5(args)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(0)
        except Exception as e:
            print(f"\nFatal error in 'train-mt5': {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    if args.command == 'live-mt5':
        try:
            _run_live_mt5(args)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(0)
        except Exception as e:
            print(f"\nFatal error in 'live-mt5': {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    # Legacy positional-mode interface
    if not args.mode:
        print("ERROR: Please specify a mode or sub-command.")
        print("  Modes: train paper live backtest dashboard")
        print("  Sub-commands: train-mt5 live-mt5")
        sys.exit(1)

    dispatch = {
        'train':     _run_train,
        'paper':     _run_paper,
        'live':      _run_live,
        'backtest':  _run_backtest,
        'dashboard': _run_dashboard,
    }

    try:
        dispatch[args.mode](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error in '{args.mode}' mode: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
