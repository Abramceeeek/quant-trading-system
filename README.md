# Quant Trading System

A **clean, staged, production-grade** quantitative trading system for authoring strategies in YAML, backtesting across universes, paper trading on IBKR, and promoting to live with strict risk guards.

## Features

- **YAML Strategy Authoring**: Define strategies declaratively without writing Python
- **Vectorized Backtesting**: Fast, robust backtests with comprehensive metrics (Sharpe, Calmar, MaxDD, turnover, etc.)
- **Universe Management**: Trade across S&P 500, custom watchlists, with liquidity/price filters
- **Walk-Forward Analysis**: Time-series CV for realistic performance estimation
- **IBKR Paper Trading**: Seamless integration with Interactive Brokers (TWS/Gateway) for paper trading
- **Risk Guards**: Global kill-switches, position limits, daily loss caps, vol ceilings
- **Live Promotion Path**: Clear workflow from backtest → paper → live with safety checks
- **Data Caching**: Smart parquet caching for yfinance; IBKR rate-limit handling
- **Modern Tooling**: Poetry, Ruff, Mypy, pytest with full type hints

## Quickstart

### 1. Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/quant-trading-system.git
cd quant-trading-system

# Install with Poetry (recommended)
poetry install

# OR with pip + venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### 2. Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` for your IBKR connection:

```bash
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # TWS Paper: 7497, Live: 7496, Gateway Paper: 4002, Live: 4001
IBKR_CLIENT_ID=9001
IBKR_ACCOUNT=DU1234567  # Your paper account ID
PAPER_TRADING=true
LIVE_TRADING=false
```

### 3. IBKR Setup (for paper/live trading)

**Full setup guide available in [IBKR_SETUP.txt](IBKR_SETUP.txt)**

**Quick setup:**
1. Download and install TWS (Trader Workstation) from IBKR
2. Enable API: File → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Uncheck "Read-Only API"
   - Port: 7497 (paper trading)
   - Add 127.0.0.1 to trusted IPs
3. Update `.env` with your paper account number (starts with DU)
4. Test connection:
```bash
python -m src.cli.papertrade strategies/breakout.yaml --dry-run
```

### 4. Run Your First Backtest

```bash
# Scan available strategies
poetry run qts scan

# Run a single strategy backtest
poetry run qts backtest strategies/sma_cross.yaml

# Walk-forward validation
poetry run qts walkforward strategies/sma_cross.yaml --folds 5

# Paper trade (dry-run)
poetry run qts papertrade strategies/sma_cross.yaml --dry-run
```

## Project Structure

```
quant-trading-system/
├── src/
│   ├── core/
│   │   ├── config.py          # Pydantic settings (env, logging, data paths)
│   │   └── universes.py       # Universe definitions (S&P500, filters, recon)
│   ├── datasource/
│   │   ├── base.py            # Abstract DataSource interface
│   │   ├── yfinance_source.py # yfinance with parquet caching
│   │   ├── ibkr_source.py     # IBKR historical data (ib_insync)
│   │   └── options_source.py  # Options chains (stub for staged rollout)
│   ├── backtest/
│   │   ├── engine.py          # Vectorized backtest engine
│   │   ├── costs.py           # Commission/slippage models
│   │   ├── sizing.py          # Position sizing (vol-target, fixed-risk)
│   │   └── metrics.py         # CAGR, Sharpe, Calmar, MaxDD, turnover, etc.
│   ├── strategies/
│   │   ├── base.py            # Abstract Strategy class
│   │   ├── sma_cross.py       # SMA crossover implementation
│   │   ├── breakout.py        # Donchian channel breakout with ATR stop
│   │   ├── earnings_iv.py     # Earnings IV short (placeholder)
│   │   └── loader.py          # YAML → Strategy parser
│   ├── execution/
│   │   ├── ibkr_client.py     # ib_insync wrapper (reconnect, error handling)
│   │   └── order_router.py    # Order placement (market, limit, bracket, sizing)
│   ├── live/
│   │   ├── risk.py            # Kill-switch, DD limits, vol ceilings
│   │   ├── portfolio_state.py # Position reconciliation
│   │   └── scheduler.py       # apscheduler for rebalance times
│   ├── cli/
│   │   ├── main.py            # Typer app entry point
│   │   ├── scan.py            # List & validate strategies
│   │   ├── backtest.py        # Run backtests
│   │   ├── walkforward.py     # Walk-forward CV
│   │   ├── papertrade.py      # Paper trading loop
│   │   └── live.py            # Live trading (guarded)
│   └── utils/
│       └── logging.py         # Logging setup
├── data/
│   ├── cache/                 # Parquet price cache
│   └── universes/             # Universe CSVs (sp500.csv, etc.)
├── runtime/
│   └── state.db               # DuckDB for executions, positions
├── reports/
│   └── YYYYMMDD/              # Backtest reports (HTML, CSV, plots)
├── strategies/
│   ├── sma_cross.yaml         # Example YAML strategies
│   └── breakout.yaml
├── tests/
│   ├── test_engine.py         # Backtest engine tests
│   ├── test_strategies.py     # Strategy logic tests
│   └── test_golden.py         # Golden file regression tests
├── pyproject.toml             # Poetry dependencies & config
├── .env.example               # Environment template
├── .gitignore
└── README.md
```

## Data Caching Rules

- **yfinance**: Auto-caches daily/intraday bars to `data/cache/*.parquet`; expires after 7 days (configurable)
- **IBKR**: Rate-limited requests with exponential backoff; historical data cached similarly
- **Offline-first**: All backtests run offline with yfinance; IBKR only for paper/live

## Strategy Authoring (YAML)

### Example: SMA Crossover

File: `strategies/sma_cross.yaml`

```yaml
strategy:
  name: sma_cross
  type: sma_crossover
  description: "Golden cross: Buy when SMA(50) > SMA(200), sell when below"

parameters:
  fast_period: 50
  slow_period: 200

universe: us_equities_core  # Defined in src/core/universes.py

backtest:
  start_date: 2020-01-01
  end_date: 2024-12-31
  initial_capital: 100000
  commission_pct: 0.001  # 10 bps per side
  slippage_pct: 0.0005   # 5 bps

sizing:
  method: equal_weight
  max_positions: 20

risk:
  stop_loss_pct: 0.05    # 5% stop
  profit_target_pct: 0.10
```

### Example: Breakout Strategy

File: `strategies/breakout.yaml`

```yaml
strategy:
  name: donchian_breakout
  type: breakout
  description: "Donchian channel breakout with ATR-based stop"

parameters:
  lookback: 20
  atr_multiplier: 2.0

universe: us_equities_liquid  # Custom liquid universe

backtest:
  start_date: 2020-01-01
  end_date: 2024-12-31
  initial_capital: 100000

sizing:
  method: vol_target
  target_vol: 0.15       # 15% annualized volatility
  max_positions: 15

risk:
  stop_loss_atr: 3.0     # 3x ATR stop
```

## Commands Reference

### Scan Strategies

```bash
poetry run qts scan
```

Lists all YAML strategies in `strategies/`, validates syntax, shows parameters.

### Backtest

```bash
# Single strategy
poetry run qts backtest strategies/sma_cross.yaml

# All strategies in folder
poetry run qts backtest strategies/

# With custom date range
poetry run qts backtest strategies/sma_cross.yaml --start 2022-01-01 --end 2023-12-31

# Generate full report (HTML + plots)
poetry run qts backtest strategies/sma_cross.yaml --report
```

Output:
- Console table: CAGR, Sharpe, Calmar, MaxDD, Turnover, Win Rate, etc.
- Reports saved to `reports/YYYYMMDD/{strategy_name}/`

### Walk-Forward Analysis

```bash
poetry run qts walkforward strategies/sma_cross.yaml --folds 5
```

Performs time-series cross-validation:
- Splits data into 5 sequential folds
- Trains on in-sample, tests on out-of-sample
- Aggregates metrics across folds
- Outputs: per-fold table + aggregate CSV

### Paper Trade

```bash
# Dry-run (no orders placed)
poetry run qts papertrade strategies/sma_cross.yaml --dry-run

# Live paper trading
poetry run qts papertrade strategies/sma_cross.yaml
```

**Flow:**
1. Connects to IBKR (paper account)
2. Fetches latest bars
3. Generates signals
4. Places orders via `order_router`
5. Logs executions to `runtime/state.db`
6. Reconciles positions daily

**Scheduling:**
Run with scheduler for automatic rebalancing:

```bash
poetry run qts papertrade strategies/sma_cross.yaml --schedule "9:35,15:50"  # 9:35 AM, 3:50 PM ET
```

### Live Trading (Production)

```bash
poetry run qts live strategies/sma_cross.yaml
```

**Requirements:**
- `LIVE_TRADING=true` in `.env`
- Risk guards enabled (see below)
- TWS/Gateway in **live mode**

**Safety Checks:**
- Confirms live mode (requires `--confirm` flag)
- Validates risk limits
- Enables kill-switch monitoring

## Risk Guards

Configured in `src/live/risk.py`:

- **Max Drawdown**: Auto-stop if portfolio drops >X% from starting equity
- **Daily Loss Limit**: No new orders if daily loss exceeds threshold
- **Position Size Clamps**: Per-symbol notional/weight limits
- **Vol Ceiling**: Reduce exposure if realized vol > target
- **Open Order Timeout**: Cancel stale orders after N minutes
- **Emergency Kill-Switch**: Manual override to flatten all positions

## Universe Management

Universes defined in `src/core/universes.py`:

### Built-in Universes

- **`us_equities_core`**: S&P 500 constituents (read from `data/universes/sp500.csv`)
- **`us_equities_liquid`**: Filtered for avg daily volume > 1M shares, price > $10
- **`demo`**: Small test list (AAPL, MSFT, SPY, QQQ, IWM)

### Adding a Custom Universe

```python
# src/core/universes.py

def get_custom_universe() -> list[str]:
    return ["AAPL", "GOOGL", "AMZN", "TSLA", "NVDA"]
```

Reference in YAML:

```yaml
universe: custom_universe
```

### Reconstitution

Universes rebalance monthly (first trading day) by default. Configurable via:

```python
# src/core/universes.py
def get_recon_dates(start: str, end: str) -> list[str]:
    # Returns list of rebalance dates (e.g., monthly, quarterly)
    ...
```

## Adding a New Strategy

### Option 1: YAML Only (Simple Rules)

Create `strategies/my_strategy.yaml`:

```yaml
strategy:
  name: my_strategy
  type: sma_crossover  # OR breakout, earnings_iv, etc.
  ...
```

The `type` maps to a Python strategy class in `src/strategies/`.

### Option 2: Custom Python Strategy

1. Create `src/strategies/my_custom.py`:

```python
from src.strategies.base import Strategy
import pandas as pd

class MyCustomStrategy(Strategy):
    def __init__(self, param1: float, param2: int):
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input: df with columns [open, high, low, close, volume]
        Output: df with columns [entry, exit, stop_loss] (boolean/float)
        """
        signals = pd.DataFrame(index=df.index)
        signals['entry'] = ...  # Your logic
        signals['exit'] = ...
        signals['stop_loss'] = ...  # Optional
        return signals
```

2. Register in `src/strategies/loader.py`:

```python
STRATEGY_REGISTRY = {
    "sma_crossover": SMACrossStrategy,
    "breakout": BreakoutStrategy,
    "my_custom": MyCustomStrategy,  # Add here
}
```

3. Reference in YAML:

```yaml
strategy:
  type: my_custom
parameters:
  param1: 0.05
  param2: 20
```

## Promoting from Paper to Live

### Workflow

1. **Backtest thoroughly**:
   ```bash
   poetry run qts backtest strategies/my_strategy.yaml --report
   poetry run qts walkforward strategies/my_strategy.yaml --folds 5
   ```

2. **Paper trade for ≥30 days**:
   ```bash
   poetry run qts papertrade strategies/my_strategy.yaml --schedule "9:35"
   ```

3. **Review paper performance**:
   ```bash
   poetry run python -c "import duckdb; duckdb.sql('SELECT * FROM runtime/state.db::executions').show()"
   ```

4. **Enable live trading**:
   - Edit `.env`: `LIVE_TRADING=true`, `PAPER_TRADING=false`
   - Update `IBKR_PORT=7496` (live TWS) or `4001` (live Gateway)
   - Set conservative risk limits in `src/live/risk.py`

5. **Run live with confirmation**:
   ```bash
   poetry run qts live strategies/my_strategy.yaml --confirm
   ```

### Risk Checklist Before Going Live

- [ ] Backtested on ≥3 years of data
- [ ] Walk-forward shows consistent out-of-sample performance
- [ ] Paper traded for ≥30 days without major issues
- [ ] Sharpe ratio ≥1.0, Calmar ≥0.5
- [ ] Max DD < 20%
- [ ] Turnover < 500% annually (unless day trading)
- [ ] Risk guards configured: max DD, daily loss, position limits
- [ ] IBKR account funded appropriately
- [ ] Kill-switch tested
- [ ] Monitoring/alerts set up

## Testing

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=src --cov-report=html

# Specific tests
poetry run pytest tests/test_engine.py -v

# Golden file test (regression)
poetry run pytest tests/test_golden.py -v
```

## Linting & Type Checking

```bash
# Ruff (linter + formatter)
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/

# Mypy (type checker)
poetry run mypy src/

# Black (formatter, optional)
poetry run black src/ tests/
```

## Development Tips

- **Use cache**: Set `CACHE_ENABLED=true` to avoid re-downloading data
- **Fast iteration**: Test on `demo` universe first (5 tickers)
- **Notebooks**: Exploratory analysis in `notebooks/` (gitignored by default)
- **Dry-run everything**: Use `--dry-run` flags liberally
- **Check logs**: `runtime/*.log` for execution details

## IBKR Connection Details

| Mode          | TWS Port | Gateway Port | Account Type |
|---------------|----------|--------------|--------------|
| Paper Trading | 7497     | 4002         | Paper (DU*)  |
| Live Trading  | 7496     | 4001         | Live         |

**Gateway vs. TWS:**
- **Gateway**: Headless, lighter, preferred for automated trading
- **TWS**: Full GUI, useful for manual monitoring

**Reconnection:**
`ibkr_client.py` handles auto-reconnect with exponential backoff (max 5 retries).

## Troubleshooting

### "No data for symbol X"

- Check ticker is valid (yfinance compatible)
- Verify date range (data may not exist for old/new tickers)
- Clear cache: `rm data/cache/*.parquet`

### "IBKR connection timeout"

- Ensure TWS/Gateway is running
- Check port (`7497` for paper, `7496` for live)
- Verify API is enabled in TWS settings
- Test: `poetry run python -c "from ib_insync import IB; ..."`

### "Strategy not found"

- Verify YAML `type` matches a registered strategy in `loader.py`
- Check YAML syntax with `poetry run qts scan`

### "Backtest returns NaN metrics"

- Likely no trades executed (check signal logic)
- Verify universe has valid data in date range
- Inspect `df` in strategy: `print(df.head())` in `generate_signals()`

## Quick Command Reference

### Backtesting
```bash
# Scan and validate all strategies
python -m src.cli.main scan strategies

# Run backtest
python -m src.cli.main backtest run strategies/sma_cross.yaml

# With custom date range
python -m src.cli.main backtest run strategies/sma_fast.yaml --start 2024-01-01

# Generate HTML report with plots
python -m src.cli.main backtest run strategies/breakout.yaml --report
```

### Paper Trading
```bash
# Dry run (preview orders without sending)
python -m src.cli.papertrade strategies/breakout.yaml --dry-run

# Send orders to IBKR paper account
python -m src.cli.papertrade strategies/breakout.yaml --no-dry-run

# With custom quantity per symbol
python -m src.cli.papertrade strategies/sma_fast.yaml --qty 50 --no-dry-run

# With notional sizing ($2000 per symbol)
python -m src.cli.papertrade strategies/sma_fast.yaml --notional 2000 --no-dry-run
```

### Available Strategies
- `strategies/sma_cross.yaml` - Golden cross (SMA 50/200) - Conservative
- `strategies/sma_fast.yaml` - Fast SMA (10/20) - More active
- `strategies/breakout.yaml` - Donchian breakout - Generates frequent signals

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

**This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly in paper mode before risking real capital. The authors assume no liability for financial losses incurred through use of this system.**

---

**Ready to build your quant trading system? Start with `python -m venv .venv && pip install -e .` and run your first backtest!**
