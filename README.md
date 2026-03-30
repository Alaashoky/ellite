# Crypto Bot Trader

A sophisticated cryptocurrency trading bot that combines Elliott Wave Theory with ICT (Inner Circle Trading) concepts to identify high-probability trading opportunities.

## Overview

This trading system uses advanced technical analysis methods including:

- **Elliott Wave Analysis**: Pattern recognition and wave counting
- **ICT Concepts**: Order blocks, fair value gaps, liquidity zones, and kill zones
- **Multi-Timeframe Analysis**: Higher timeframe bias with lower timeframe precision entries
- **Regime-Adaptive Scoring**: Dynamic adjustment based on market conditions
- **Market Structure Detection**: Break of structure (BOS) and change of character (CHoCH) identification

## Features

- Multi-timeframe signal generation (HTF bias + LTF entries)
- Advanced risk management with dynamic position sizing
- Comprehensive backtesting engine with walk-forward analysis
- Production validation suite for pre-deployment testing
- Fibonacci-based stop loss and take profit levels
- Kill zone awareness (London, New York sessions)
- Wave ranking system for trade prioritization
- Multi-confirmation entry system

## Project Structure

```
.
├── backtest.py                      # Unified backtesting interface
├── backtester.py                    # Core backtesting engine
├── config/                          # Configuration files
│   ├── pairs.yaml                   # Trading pairs configuration
│   ├── timeframes.yaml              # Timeframe settings
│   └── trading_config.yaml          # Main trading strategy config
├── trading_strategy/                # Core trading strategy modules
│   ├── config_loader.py             # Configuration management
│   ├── data_loader.py               # Data loading and caching
│   ├── data_structures.py           # Signal and trade data structures
│   ├── elliott_wave.py              # Elliott Wave detection
│   ├── ict_concepts.py              # ICT concepts implementation
│   ├── ict_entries.py               # ICT-based entry signals
│   ├── kill_zones.py                # Trading session detection
│   ├── ltf_precision_entry.py       # Lower timeframe entry logic
│   ├── market_structure.py          # Market structure analysis
│   ├── regime_adaptive_scoring.py   # Regime-based signal scoring
│   └── trading_strategy.py          # Main strategy orchestration
```

## Configuration

The strategy is configured through YAML files in the `config/` directory:

### trading_config.yaml

Contains strategy parameters including:

- Elliott Wave settings
- ICT concept parameters
- Risk management rules
- Entry confirmation requirements
- Wave ranking criteria

### timeframes.yaml

Defines the multi-timeframe structure:

- Higher timeframe (HTF) for bias
- Medium timeframe (MTF) for structure
- Lower timeframe (LTF) for entries

### pairs.yaml

Lists the trading pairs to analyze

## Strategy Logic

### Signal Generation Process

1. **HTF Bias Determination**: Analyze higher timeframe for overall market direction
2. **Market Structure**: Identify key support/resistance levels and structure breaks
3. **Elliott Wave Analysis**: Detect wave patterns and count wave positions
4. **ICT Concepts**: Identify order blocks, fair value gaps, and liquidity zones
5. **Kill Zone Filter**: Check if current time is within optimal trading sessions
6. **LTF Entry Confirmation**: Look for precise entry triggers on lower timeframe
7. **Multi-Confirmation**: Require multiple confirmations before generating signal
8. **Wave Ranking**: Score and prioritize signals based on quality

### Risk Management

- Dynamic position sizing based on account balance
- Maximum risk per trade (configurable, typically 1-2%)
- Maximum daily risk limits
- Maximum concurrent positions
- Fibonacci-based stop loss placement
- Multiple take profit targets with partial exits

## Backtesting

The backtesting engine provides:

- Realistic execution modeling
- Slippage and commission simulation
- Multiple position management
- Partial profit taking
- Comprehensive performance metrics
- Trade-by-trade analysis
- Equity curve generation
- Drawdown analysis

### Backtest Metrics

- Total return and annualized return
- Win rate and profit factor
- Average win/loss ratio
- Maximum drawdown
- Sharpe ratio
- Recovery factor
- Trade statistics (count, duration, etc.)

---

## Separated Train / Backtest Workflows

To keep model training and out-of-sample evaluation cleanly separated, two
dedicated scripts are provided:

| Script | Purpose | Default window |
|--------|---------|----------------|
| `train_model.py` | Train AI models on **in-sample** historical data | 2020-01-01 → 2023-12-31 |
| `backtest_oos.py` | Evaluate trained models on **out-of-sample** data | 2024-01-01 → today |

### Training (`train_model.py`)

```bash
# Train with defaults (2020-01-01 → 2023-12-31, BTCUSDT, 15m):
python train_model.py

# Train a different symbol on a custom date range:
python train_model.py --symbol XAUUSDT --start-date 2019-01-01 --end-date 2023-12-31

# Train using MT5 as data source:
python train_model.py --symbol XAUUSDT --data-source mt5 --mt5-symbol XAUUSD.m

# Train with walk-forward validation (5 folds):
python train_model.py --symbol BTCUSDT --walk-forward --n-splits 5
```

Trained model artifacts are saved to `saved_models/` (override with `--models-dir`).
A log file is written to `logs/train_model_<SYMBOL>_<TIMESTAMP>.log` (override with `--logs-dir`).

### Backtesting (`backtest_oos.py`)

```bash
# Run backtest with defaults (2024-01-01 → today, BTCUSDT, single mode):
python backtest_oos.py

# Backtest a specific symbol on a custom out-of-sample period:
python backtest_oos.py --symbol XAUUSD --start-date 2024-01-01 --end-date 2025-01-01

# Quick validation with a larger initial balance:
python backtest_oos.py --mode quick --initial-balance 50000

# Walk-forward analysis:
python backtest_oos.py --symbol BTCUSDT --mode walkforward

# Custom report output directory:
python backtest_oos.py --report-dir ./my_reports
```

JSON reports are saved to `backtest_reports/` (override with `--report-dir`).
Each report is timestamped: `backtest_oos_<SYMBOL>_<MODE>_<TIMESTAMP>.json`.
Log files are written to `logs/backtest_oos_<SYMBOL>_<TIMESTAMP>.log`.

### CLI quick-reference

| Argument | `train_model.py` | `backtest_oos.py` |
|---|---|---|
| `--symbol` | ✔ (BTCUSDT / XAUUSDT) | ✔ (any symbol) |
| `--mt5-symbol` | ✔ | ✔ |
| `--start-date` | default `2020-01-01` | default `2024-01-01` |
| `--end-date` | default `2023-12-31` | default today |
| `--timeframe` | ✔ | — |
| `--data-source` | ccxt / mt5 | — |
| `--models-dir` | ✔ | ✔ |
| `--logs-dir` | ✔ | ✔ |
| `--walk-forward` / `--n-splits` | ✔ | — |
| `--initial-balance` | — | ✔ |
| `--mode` | — | quick / single / multi / walkforward |
| `--report-dir` | — | ✔ |

> **CLI overrides config**: date ranges passed on the command line always take
> precedence over any defaults in `config/` files.

---

## MT5 Setup (MetaTrader 5)

The bot supports MetaTrader 5 as an alternative data source and execution engine,
giving access to Forex, Gold (XAUUSD), and Crypto CFDs through any MT5 broker.

> **⚠️ Windows only** — The MetaTrader5 Python API only runs on Windows.
> On Linux/macOS a clear `ImportError` with a helpful message is raised.

### 1. Install MetaTrader 5 Terminal

Download and install the [MetaTrader 5 terminal](https://www.metatrader5.com/en/download)
on Windows, then log in with your broker account.

### 2. Configure credentials

Either fill in `config/mt5_config.yaml`:

```yaml
mt5:
  login: 123456              # your MT5 account number
  password: "YourPassword"
  server: "ICMarkets-Live01" # your broker's server name
  path: ""                   # leave empty unless terminal is non-default location
```

Or set environment variables (these take priority over the config file):

```bash
MT5_LOGIN=123456
MT5_PASSWORD=YourPassword
MT5_SERVER=ICMarkets-Live01
MT5_PATH=
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Train AI models on MT5 historical data

Fetch data from 2020-01-01 to 2023-12-31 and train all models:

```bash
python main.py train-mt5 --symbols EURUSD,XAUUSD --start 2020-01-01 --end 2023-12-31 --timeframe 15m
```

You can train on multiple symbols by passing a comma-separated list:

```bash
python main.py train-mt5 --symbols EURUSD,GBPUSD,XAUUSD,BTCUSD --start 2020-01-01 --end 2023-12-31
```

Trained models are saved to `saved_models/`.

### 5. Run the MT5 trading bot

**Paper mode** (no real orders, full simulation):

```bash
python main.py live-mt5 --symbols EURUSD,XAUUSD --mode paper
```

**Live mode** (real orders sent to MT5 terminal):

```bash
python main.py live-mt5 --symbols EURUSD,XAUUSD --mode live
```

### CLI reference

| Sub-command | Key options | Description |
|-------------|-------------|-------------|
| `train-mt5` | `--symbols`, `--start`, `--end`, `--timeframe` | Train models from MT5 data |
| `live-mt5`  | `--symbols`, `--mode paper\|live`, `--balance` | Start MT5 trading loop |

### Notes

- Data is fetched in **year-long chunks** (`copy_rates_range` batching) to reliably retrieve all history from 2020 regardless of broker candle limits.
- All timestamps are normalised to **UTC**.
- The existing Binance/ccxt workflow (`train`, `paper`, `live`, `backtest`) remains **100 % unchanged**.
- MT5 symbol names differ by broker — use the exact name shown in your MT5 Market Watch (e.g. `EURUSD`, `XAUUSD`, `BTCUSD`).
