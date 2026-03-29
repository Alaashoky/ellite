"""Live and paper trading loop with rule-based + AI signal combination."""

import logging
import time
from typing import Dict, List, Optional

import schedule

from exchange_connector import ExchangeConnector
from risk_manager import RiskManager
from order_manager import OrderManager
from portfolio_tracker import PortfolioTracker
from notifications import TelegramNotifier
from logger import setup_logger, log_trade
from ai_models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Main trading loop.

    Runs on a 15-minute schedule, fetches OHLCV data, generates rule-based
    and AI signals, validates risk, and executes bracket orders.
    """

    def __init__(
        self,
        assets: List[str],
        model_type: str = 'ensemble',
        mode: str = 'paper',
        initial_balance: float = 10_000.0,
    ):
        self.assets = assets
        self.model_type = model_type
        self.mode = mode
        self.initial_balance = initial_balance
        self._running = False
        self._components_ready = False
        setup_logger()
        self._setup_components()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_components(self) -> None:
        """Initialise all sub-components."""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        paper_mode = self.mode != 'live'

        self.exchange = ExchangeConnector(
            api_key=api_key,
            secret_key=secret_key,
            paper_mode=paper_mode,
            initial_balance=self.initial_balance,
        )
        self.risk_manager = RiskManager(
            initial_balance=self.initial_balance,
            risk_per_trade=float(os.getenv('RISK_PER_TRADE', '0.01')),
            max_daily_risk=float(os.getenv('MAX_DAILY_RISK', '0.05')),
        )
        self.order_manager = OrderManager(self.exchange, self.risk_manager)
        self.portfolio = PortfolioTracker(
            initial_balance=self.initial_balance,
            state_file='./data/portfolio_state.json',
        )

        tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        tg_chat = os.getenv('TELEGRAM_CHAT_ID')
        self.notifier = TelegramNotifier(bot_token=tg_token, chat_id=tg_chat)

        # Load AI models
        models_dir = os.getenv('MODELS_PATH', './saved_models')
        self.ai_models: Dict[str, EnsembleModel] = {}
        for symbol in self.assets:
            em = EnsembleModel(symbol, models_dir)
            try:
                em.load_models()
            except Exception as e:
                logger.warning(f"Could not load AI models for {symbol}: {e}")
            self.ai_models[symbol] = em

        # Rule-based strategy (optional – graceful degradation)
        try:
            from trading_strategy import TradingStrategy
            self.strategies: Dict[str, object] = {
                sym: TradingStrategy(sym) for sym in self.assets
            }
        except (ImportError, ModuleNotFoundError):
            self.strategies = {}
            logger.warning("TradingStrategy not available; rule-based signals disabled.")

        self._components_ready = True
        logger.info(f"LiveTrader ready — mode={self.mode}, assets={self.assets}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the scheduler and block until stop() is called."""
        self._running = True
        self.notifier.send_startup(self.assets, self.mode)

        schedule.every(15).minutes.do(self._trading_cycle)
        schedule.every().day.at('00:01').do(self._daily_reset)

        logger.info("Scheduler started. Running first cycle immediately.")
        self._trading_cycle()

        while self._running:
            schedule.run_pending()
            time.sleep(1)

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Stopping LiveTrader…")
        self._running = False
        self.portfolio.save_state()
        logger.info("Portfolio state saved. Goodbye.")

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    def _trading_cycle(self) -> None:
        """Called every 15 minutes: update prices and process each asset."""
        try:
            prices = {}
            for symbol in self.assets:
                try:
                    ticker = self.exchange.get_ticker(symbol)
                    if ticker:
                        prices[symbol] = float(ticker.get('last', 0))
                except Exception as e:
                    logger.warning(f"Ticker fetch failed for {symbol}: {e}")

            if prices:
                self.portfolio.update_prices(prices)

            for symbol in self.assets:
                try:
                    self._process_asset(symbol)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                    self.notifier.send_error_alert(f"{symbol}: {e}")
        except Exception as e:
            logger.error(f"Trading cycle error: {e}", exc_info=True)

    def _process_asset(self, symbol: str) -> None:
        """
        Full pipeline for one asset:
        1. Fetch OHLCV
        2. Manage existing positions
        3. Rule-based signal
        4. AI signal
        5. Combine signals
        6. Validate risk
        7. Size position
        8. Execute bracket order
        9. Notify & log
        """
        # 1. Fetch data
        df = self.exchange.get_ohlcv(symbol, timeframe='15m', limit=300)
        if df is None or len(df) < 100:
            logger.warning(f"{symbol}: insufficient OHLCV data ({len(df) if df is not None else 0} rows)")
            return

        current_price = float(df['close'].iloc[-1])

        # 2. Manage existing position
        self._manage_positions(symbol, current_price)

        # Skip if already in a position
        if symbol in self.portfolio.open_positions:
            return

        # 3. Rule-based signal
        rule_signal: Optional[str] = None
        strategy = self.strategies.get(symbol)
        if strategy:
            try:
                rule_signal = strategy.get_signal(df)
            except Exception as e:
                logger.debug(f"Rule signal error {symbol}: {e}")

        # 4. AI signal
        ai_signal: Optional[str] = None
        ai_confidence: float = 0.0
        em = self.ai_models.get(symbol)
        if em:
            try:
                ai_signal, ai_confidence, _ = em.get_signal(df)
            except Exception as e:
                logger.debug(f"AI signal error {symbol}: {e}")

        # 5. Combine signals
        final_signal = self._combine_signals(rule_signal, ai_signal, ai_confidence)
        if final_signal is None or final_signal == 'HOLD':
            return

        # 6. Risk validation
        allowed, reason = self.risk_manager.validate_trade(final_signal, ai_confidence)
        if not allowed:
            logger.info(f"{symbol}: trade blocked — {reason}")
            return

        # 7. Position sizing via ATR-based SL
        from config.settings import ASSETS_CONFIG
        cfg = ASSETS_CONFIG.get(symbol, {})
        try:
            from ai_models.feature_engineering import FeatureEngineer
            fe = FeatureEngineer(symbol)
            df_feat = fe.create_features(df)
            atr = float(df_feat['atr_14'].iloc[-1])
        except Exception:
            atr = current_price * 0.01

        atr_sl_mult = cfg.get('atr_sl_multiplier', 2.0)
        atr_tp_mult = cfg.get('atr_tp_multiplier', 4.0)

        if final_signal == 'BUY':
            stop_loss = current_price - atr * atr_sl_mult
            take_profits = [
                current_price + atr * atr_tp_mult * 0.5,
                current_price + atr * atr_tp_mult,
            ]
        else:  # SELL
            stop_loss = current_price + atr * atr_sl_mult
            take_profits = [
                current_price - atr * atr_tp_mult * 0.5,
                current_price - atr * atr_tp_mult,
            ]

        amount = self.risk_manager.calculate_position_size(current_price, stop_loss, symbol)
        if amount <= 0:
            logger.warning(f"{symbol}: calculated amount is 0, skipping.")
            return

        # 8. Execute bracket order
        side = 'buy' if final_signal == 'BUY' else 'sell'
        try:
            bracket = self.order_manager.place_bracket_order(
                symbol, side, amount, current_price, stop_loss, take_profits
            )
        except Exception as e:
            logger.error(f"Failed to place bracket order for {symbol}: {e}")
            return

        # Register position
        self.portfolio.open_position(
            symbol, side, current_price, amount, stop_loss, take_profits,
            model_signal=f'ai:{ai_signal}/rule:{rule_signal}',
            confidence=ai_confidence,
        )
        self.risk_manager.add_position(symbol, bracket)

        # 9. Notify & log
        trade_data = {
            'symbol': symbol,
            'side': side,
            'entry_price': current_price,
            'amount': amount,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'confidence': ai_confidence,
            'mode': self.mode,
        }
        self.notifier.send_trade_alert(trade_data)
        log_trade(trade_data)
        logger.info(
            f"[{self.mode.upper()}] {final_signal} {symbol} @ {current_price:.4f} "
            f"amount={amount} SL={stop_loss:.4f} conf={ai_confidence:.2%}"
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_positions(self, symbol: str, current_price: float) -> None:
        """Check SL/TP triggers for an open position and close if hit."""
        pos = self.portfolio.open_positions.get(symbol)
        if pos is None:
            return

        sl = pos.get('stop_loss')
        tps = pos.get('take_profits', [])
        side = pos.get('side', 'buy')

        hit_sl = (
            (side == 'buy' and current_price <= sl) or
            (side == 'sell' and current_price >= sl)
        ) if sl else False

        hit_tp = False
        if tps:
            first_tp = tps[0]
            hit_tp = (
                (side == 'buy' and current_price >= first_tp) or
                (side == 'sell' and current_price <= first_tp)
            )

        if hit_sl:
            trade = self.portfolio.close_position(symbol, current_price, reason='stop_loss')
            if trade:
                self.risk_manager.update_balance(trade['pnl'])
                self.risk_manager.remove_position(symbol)
                self.notifier.send_trade_alert({**trade, 'reason': 'SL hit'})
                log_trade(trade)
                logger.info(f"SL hit for {symbol} @ {current_price:.4f} PnL={trade['pnl']:.2f}")

        elif hit_tp and len(tps) > 1:
            # Partial close on first TP
            trade = self.portfolio.partial_close(symbol, current_price, close_fraction=0.5)
            if trade:
                self.risk_manager.update_balance(trade['pnl'])
                logger.info(f"Partial TP hit for {symbol} @ {current_price:.4f} PnL={trade['pnl']:.2f}")
                if pos.get('take_profits'):
                    pos['take_profits'] = pos['take_profits'][1:]

        elif hit_tp:
            trade = self.portfolio.close_position(symbol, current_price, reason='take_profit')
            if trade:
                self.risk_manager.update_balance(trade['pnl'])
                self.risk_manager.remove_position(symbol)
                self.notifier.send_trade_alert({**trade, 'reason': 'TP hit'})
                log_trade(trade)
                logger.info(f"TP hit for {symbol} @ {current_price:.4f} PnL={trade['pnl']:.2f}")

    # ------------------------------------------------------------------
    # Signal combination
    # ------------------------------------------------------------------

    def _combine_signals(
        self,
        rule_signal: Optional[str],
        ai_signal: Optional[str],
        ai_confidence: float,
    ) -> Optional[str]:
        """
        Signal combination logic:
        - If AI confidence >= 0.75 → use AI signal alone.
        - Elif rule_signal == ai_signal (and both are BUY/SELL) → use that signal.
        - Else → None (skip trade).
        """
        if ai_signal and ai_confidence >= 0.75:
            return ai_signal
        if rule_signal and ai_signal and rule_signal == ai_signal and rule_signal in ('BUY', 'SELL'):
            return rule_signal
        return None

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _daily_reset(self) -> None:
        """Reset daily counters, send report, persist state."""
        self.risk_manager.reset_daily_pnl()
        report = self.portfolio.get_daily_report()
        self.notifier.send_daily_report(report)
        self.portfolio.save_state()
        logger.info(f"Daily reset. Report: {report}")
