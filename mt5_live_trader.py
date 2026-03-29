"""
mt5_live_trader.py

Live/paper trading loop backed by MetaTrader 5.
Mirrors the existing LiveTrader API but uses MT5Connector for execution
and MT5DataLoader for data retrieval.
"""

import logging
import os
import time
from typing import Dict, List, Optional

import schedule

from mt5_connector import MT5Connector
from trading_strategy.mt5_data_loader import MT5DataLoader
from risk_manager import RiskManager
from portfolio_tracker import PortfolioTracker
from notifications import TelegramNotifier
from logger import setup_logger, log_trade
from ai_models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


def _load_mt5_connector() -> MT5Connector:
    """
    Build an MT5Connector from environment variables or config file.

    Priority:
        1. Environment variables: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH
        2. config/mt5_config.yaml
    """
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH", "")

    if not (login and password and server):
        try:
            import yaml  # type: ignore
            config_path = os.path.join(
                os.path.dirname(__file__), "config", "mt5_config.yaml"
            )
            with open(config_path, "r") as fh:
                cfg = yaml.safe_load(fh)
            mt5_cfg = cfg.get("mt5", {})
            login = login or str(mt5_cfg.get("login", "0"))
            password = password or str(mt5_cfg.get("password", ""))
            server = server or str(mt5_cfg.get("server", "MetaQuotes-Demo"))
            path = path or str(mt5_cfg.get("path", ""))
        except Exception as exc:
            logger.warning(f"Could not load mt5_config.yaml: {exc}")

    if not login or not password or not server:
        raise RuntimeError(
            "MT5 credentials are not configured. "
            "Set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER environment variables "
            "or fill in config/mt5_config.yaml."
        )

    return MT5Connector(
        login=int(login),
        password=password,
        server=server,
        path=path,
    )


class MT5LiveTrader:
    """
    Main trading loop backed by MetaTrader 5.

    Runs on a 15-minute schedule, fetches the latest candle via MT5DataLoader,
    generates rule-based and AI signals, validates risk, and:
      - In 'paper' mode  : simulates trades without sending real orders.
      - In 'live'  mode  : executes real orders via MT5Connector.place_order().

    Constructor parameters are intentionally identical to LiveTrader so that
    callers can switch between the two with minimal changes.
    """

    def __init__(
        self,
        assets: List[str],
        model_type: str = "ensemble",
        mode: str = "paper",
        initial_balance: float = 10_000.0,
    ) -> None:
        self.assets = assets
        self.model_type = model_type
        self.mode = mode
        self.initial_balance = initial_balance
        self._running = False
        setup_logger()
        self._setup_components()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_components(self) -> None:
        """Initialise all sub-components."""
        from dotenv import load_dotenv
        load_dotenv()

        # MT5 connector + data loader
        self.connector = _load_mt5_connector()
        if not self.connector.connect():
            raise RuntimeError("Failed to connect to MetaTrader 5 terminal.")
        self.data_loader = MT5DataLoader(self.connector)

        # Risk / portfolio
        self.risk_manager = RiskManager(
            initial_balance=self.initial_balance,
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
            max_daily_risk=float(os.getenv("MAX_DAILY_RISK", "0.05")),
        )
        self.portfolio = PortfolioTracker(
            initial_balance=self.initial_balance,
            state_file="./data/mt5_portfolio_state.json",
        )

        # Notifications
        tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        self.notifier = TelegramNotifier(bot_token=tg_token, chat_id=tg_chat)

        # AI models
        models_dir = os.getenv("MODELS_PATH", "./saved_models")
        self.ai_models: Dict[str, EnsembleModel] = {}
        for symbol in self.assets:
            em = EnsembleModel(symbol, models_dir)
            try:
                em.load_models()
            except Exception as exc:
                logger.warning(f"Could not load AI models for {symbol}: {exc}")
            self.ai_models[symbol] = em

        # Rule-based strategy (optional — graceful degradation)
        try:
            from trading_strategy import TradingStrategy
            self.strategies: Dict[str, object] = {
                sym: TradingStrategy(sym) for sym in self.assets
            }
        except (ImportError, ModuleNotFoundError):
            self.strategies = {}
            logger.warning(
                "TradingStrategy not available; rule-based signals disabled."
            )

        logger.info(
            f"MT5LiveTrader ready — mode={self.mode}, assets={self.assets}"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the scheduler and block until stop() is called."""
        self._running = True
        self.notifier.send_startup(self.assets, self.mode)

        schedule.every(15).minutes.do(self._trading_cycle)
        schedule.every().day.at("00:01").do(self._daily_reset)

        logger.info("MT5 scheduler started. Running first cycle immediately.")
        self._trading_cycle()

        while self._running:
            schedule.run_pending()
            time.sleep(1)

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Stopping MT5LiveTrader…")
        self._running = False
        self.portfolio.save_state()
        self.connector.disconnect()
        logger.info("Portfolio state saved. MT5 disconnected. Goodbye.")

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    def _trading_cycle(self) -> None:
        """Called every 15 minutes: fetch prices and process each asset."""
        try:
            for symbol in self.assets:
                try:
                    self._process_asset(symbol)
                except Exception as exc:
                    logger.error(
                        f"Error processing {symbol}: {exc}", exc_info=True
                    )
                    self.notifier.send_error_alert(f"{symbol}: {exc}")
        except Exception as exc:
            logger.error(f"Trading cycle error: {exc}", exc_info=True)

    def _process_asset(self, symbol: str) -> None:
        """
        Full pipeline for one asset:
        1. Fetch latest OHLCV candles
        2. Manage existing positions
        3. Rule-based signal
        4. AI signal
        5. Combine signals
        6. Validate risk
        7. Size position
        8. Execute order (real via MT5 in 'live' mode, simulated in 'paper' mode)
        9. Notify & log
        """
        # 1. Fetch recent candles (300 bars should be plenty for indicators)
        try:
            df = self.data_loader.load_data(
                symbol=symbol,
                timeframe="15m",
                # Let MT5DataLoader default to "2020-01-01" as start; we then
                # trim to the last 300 rows so we only use recent data.
            )
            # Trim to the last 300 rows for efficiency
            df = df.tail(300).copy()
        except Exception as exc:
            logger.warning(f"{symbol}: failed to load data — {exc}")
            return

        if len(df) < 100:
            logger.warning(
                f"{symbol}: insufficient data ({len(df)} rows), skipping."
            )
            return

        current_price = float(df["close"].iloc[-1])

        # 2. Manage existing position
        self._manage_positions(symbol, current_price)

        if symbol in self.portfolio.open_positions:
            return

        # 3. Rule-based signal
        rule_signal: Optional[str] = None
        strategy = self.strategies.get(symbol)
        if strategy:
            try:
                rule_signal = strategy.get_signal(df)
            except Exception as exc:
                logger.debug(f"Rule signal error {symbol}: {exc}")

        # 4. AI signal
        ai_signal: Optional[str] = None
        ai_confidence: float = 0.0
        em = self.ai_models.get(symbol)
        if em:
            try:
                ai_signal, ai_confidence, _ = em.get_signal(df)
            except Exception as exc:
                logger.debug(f"AI signal error {symbol}: {exc}")

        # 5. Combine signals
        final_signal = self._combine_signals(rule_signal, ai_signal, ai_confidence)
        if final_signal is None or final_signal == "HOLD":
            return

        # 6. Risk validation
        allowed, reason = self.risk_manager.validate_trade(final_signal, ai_confidence)
        if not allowed:
            logger.info(f"{symbol}: trade blocked — {reason}")
            return

        # 7. Position sizing via ATR-based SL
        try:
            from ai_models.feature_engineering import FeatureEngineer
            fe = FeatureEngineer(symbol)
            df_feat = fe.create_features(df)
            atr = float(df_feat["atr_14"].iloc[-1])
        except Exception:
            atr = current_price * 0.01

        atr_sl_mult = 2.0
        atr_tp_mult = 4.0

        if final_signal == "BUY":
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

        volume = self.risk_manager.calculate_position_size(
            current_price, stop_loss, symbol
        )
        if volume <= 0:
            logger.warning(f"{symbol}: calculated volume is 0, skipping.")
            return

        # 8. Execute order
        side = "buy" if final_signal == "BUY" else "sell"
        order_result: Optional[Dict] = None

        if self.mode == "live":
            try:
                order_result = self.connector.place_order(
                    symbol=symbol,
                    order_type=side,
                    volume=volume,
                    sl=stop_loss,
                    tp=take_profits[-1],
                    comment="ellite_bot",
                )
            except Exception as exc:
                logger.error(f"Failed to place MT5 order for {symbol}: {exc}")
                return
        else:
            # Paper mode — simulate without sending a real order
            order_result = {
                "retcode": 10009,  # TRADE_RETCODE_DONE equivalent
                "order": 0,
                "price": current_price,
                "comment": "paper_trade",
            }
            logger.info(
                f"[PAPER] Simulated {side.upper()} {symbol} "
                f"@ {current_price:.5f} vol={volume}"
            )

        # Register position in portfolio tracker
        self.portfolio.open_position(
            symbol, side, current_price, volume, stop_loss, take_profits,
            model_signal=f"ai:{ai_signal}/rule:{rule_signal}",
            confidence=ai_confidence,
        )
        self.risk_manager.add_position(symbol, order_result or {})

        # 9. Notify & log
        trade_data = {
            "symbol": symbol,
            "side": side,
            "entry_price": current_price,
            "amount": volume,
            "stop_loss": stop_loss,
            "take_profits": take_profits,
            "confidence": ai_confidence,
            "mode": self.mode,
        }
        self.notifier.send_trade_alert(trade_data)
        log_trade(trade_data)
        logger.info(
            f"[{self.mode.upper()}] {final_signal} {symbol} "
            f"@ {current_price:.5f} vol={volume} SL={stop_loss:.5f} "
            f"conf={ai_confidence:.2%}"
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_positions(self, symbol: str, current_price: float) -> None:
        """Check SL/TP triggers for an open position and close if hit."""
        pos = self.portfolio.open_positions.get(symbol)
        if pos is None:
            return

        sl = pos.get("stop_loss")
        tps = pos.get("take_profits", [])
        side = pos.get("side", "buy")

        hit_sl = (
            (side == "buy" and current_price <= sl) or
            (side == "sell" and current_price >= sl)
        ) if sl else False

        hit_tp = False
        if tps:
            first_tp = tps[0]
            hit_tp = (
                (side == "buy" and current_price >= first_tp) or
                (side == "sell" and current_price <= first_tp)
            )

        if hit_sl:
            # In live mode, close MT5 position by ticket if available
            if self.mode == "live":
                ticket = pos.get("ticket")
                if ticket:
                    self.connector.close_position(ticket)

            trade = self.portfolio.close_position(symbol, current_price, reason="stop_loss")
            if trade:
                self.risk_manager.update_balance(trade["pnl"])
                self.risk_manager.remove_position(symbol)
                self.notifier.send_trade_alert({**trade, "reason": "SL hit"})
                log_trade(trade)
                logger.info(
                    f"SL hit for {symbol} @ {current_price:.5f} PnL={trade['pnl']:.2f}"
                )

        elif hit_tp and len(tps) > 1:
            trade = self.portfolio.partial_close(symbol, current_price, close_fraction=0.5)
            if trade:
                self.risk_manager.update_balance(trade["pnl"])
                logger.info(
                    f"Partial TP hit for {symbol} @ {current_price:.5f} "
                    f"PnL={trade['pnl']:.2f}"
                )
                if pos.get("take_profits"):
                    pos["take_profits"] = pos["take_profits"][1:]

        elif hit_tp:
            if self.mode == "live":
                ticket = pos.get("ticket")
                if ticket:
                    self.connector.close_position(ticket)

            trade = self.portfolio.close_position(symbol, current_price, reason="take_profit")
            if trade:
                self.risk_manager.update_balance(trade["pnl"])
                self.risk_manager.remove_position(symbol)
                self.notifier.send_trade_alert({**trade, "reason": "TP hit"})
                log_trade(trade)
                logger.info(
                    f"TP hit for {symbol} @ {current_price:.5f} PnL={trade['pnl']:.2f}"
                )

    # ------------------------------------------------------------------
    # Signal combination
    # ------------------------------------------------------------------

    def _combine_signals(
        self,
        rule_signal: Optional[str],
        ai_signal: Optional[str],
        ai_confidence: float,
    ) -> Optional[str]:
        """Mirror the signal combination logic from LiveTrader."""
        if ai_signal and ai_confidence >= 0.75:
            return ai_signal
        if (
            rule_signal and ai_signal
            and rule_signal == ai_signal
            and rule_signal in ("BUY", "SELL")
        ):
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
