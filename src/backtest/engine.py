"""
Vectorized backtest engine.
Orchestrates data loading, signal generation, position sizing, costs/slippage, and metrics.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.backtest.costs import get_cost_model
from src.backtest.metrics import calculate_all_metrics, format_metrics
from src.backtest.sizing import get_sizing_method
from src.backtest.slippage import get_slippage_model
from src.datasource.base import DataSource

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Vectorized backtest engine for equity strategies."""

    def __init__(
        self,
        datasource: DataSource,
        initial_capital: float = 100_000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        sizing_method: str = "equal_weight",
        max_positions: int = 20,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            datasource: DataSource instance for fetching prices
            initial_capital: Starting capital
            commission_pct: Commission as decimal
            slippage_pct: Slippage as decimal
            sizing_method: 'equal_weight', 'vol_target', 'fixed_risk', 'kelly'
            max_positions: Maximum simultaneous positions
        """
        self.datasource = datasource
        self.initial_capital = initial_capital
        self.max_positions = max_positions

        # Models
        self.cost_model = get_cost_model("percentage", commission_pct=commission_pct)
        self.slippage_model = get_slippage_model("fixed", slippage_pct=slippage_pct)
        self.sizing_func = get_sizing_method(sizing_method)

        # Results storage
        self.equity_curve: pd.Series | None = None
        self.returns: pd.Series | None = None
        self.trades: pd.DataFrame | None = None
        self.positions: pd.DataFrame | None = None
        self.metrics: dict | None = None

    def run(
        self,
        strategy: Any,
        universe: list[str],
        start: str,
        end: str,
        **sizing_kwargs: Any,
    ) -> dict:
        """
        Run backtest.

        Args:
            strategy: Strategy instance with generate_signals(df) method
            universe: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            **sizing_kwargs: Additional kwargs for sizing function

        Returns:
            Dict of backtest results (metrics, equity_curve, trades, etc.)
        """
        logger.info(f"Running backtest: {len(universe)} symbols, {start} to {end}")

        # Load price data
        price_data = self.datasource.get_prices(universe, start, end, interval="1d")

        if not price_data:
            raise ValueError("No price data loaded. Check symbols and date range.")

        # Align data to common date range
        aligned_data = self._align_price_data(price_data)
        dates = aligned_data.index

        logger.info(f"Loaded {len(aligned_data)} trading days for {len(price_data)} symbols")

        # Generate signals for each symbol
        all_signals = {}
        for symbol, df in price_data.items():
            try:
                signals = strategy.generate_signals(df)
                all_signals[symbol] = signals
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")

        if not all_signals:
            raise ValueError("No signals generated. Check strategy logic.")

        # Run simulation
        self._simulate(aligned_data, all_signals, **sizing_kwargs)

        # Calculate metrics
        self.metrics = calculate_all_metrics(self.equity_curve, self.returns, self.trades)

        logger.info("Backtest complete")
        return {
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
            "returns": self.returns,
            "trades": self.trades,
            "positions": self.positions,
        }

    def _align_price_data(self, price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align price data to a common date index.

        Args:
            price_data: Dict of {symbol: DataFrame}

        Returns:
            DataFrame with aligned close prices (columns = symbols, index = dates)
        """
        close_prices = {}
        for symbol, df in price_data.items():
            if "close" in df.columns:
                close_prices[symbol] = df["close"]

        aligned = pd.DataFrame(close_prices)
        aligned = aligned.dropna(how="all")  # Remove dates with no data
        return aligned

    def _simulate(
        self,
        prices: pd.DataFrame,
        signals: dict[str, pd.DataFrame],
        **sizing_kwargs: Any,
    ) -> None:
        """
        Run the actual simulation loop.

        Args:
            prices: Aligned price DataFrame (dates x symbols)
            signals: Dict of {symbol: signals DataFrame}
            **sizing_kwargs: Sizing parameters
        """
        dates = prices.index
        symbols = prices.columns.tolist()

        # Initialize state
        cash = self.initial_capital
        positions = pd.Series(0, index=symbols)  # Current shares held
        equity_curve = []
        trade_log = []

        for i, date in enumerate(dates):
            current_prices = prices.loc[date]

            # Calculate current portfolio value
            position_value = (positions * current_prices).sum()
            equity = cash + position_value
            equity_curve.append(equity)

            # Generate entry/exit decisions for today
            entries = pd.Series(False, index=symbols)
            exits = pd.Series(False, index=symbols)

            for symbol in symbols:
                if symbol not in signals:
                    continue

                sig = signals[symbol]
                if date not in sig.index:
                    continue

                if "entry" in sig.columns:
                    entries[symbol] = bool(sig.loc[date, "entry"])
                if "exit" in sig.columns:
                    exits[symbol] = bool(sig.loc[date, "exit"])

            # Process exits first (flatten existing positions)
            for symbol in symbols:
                if exits[symbol] and positions[symbol] != 0:
                    shares_to_sell = positions[symbol]
                    exit_price = current_prices[symbol]

                    # Apply slippage
                    trade_df = pd.DataFrame({"shares": [-shares_to_sell]}, index=[symbol])
                    execution_price = self.slippage_model.apply_slippage(
                        trade_df, pd.Series([exit_price], index=[symbol])
                    )[symbol]

                    # Calculate proceeds
                    proceeds = shares_to_sell * execution_price

                    # Apply costs
                    costs = self.cost_model.calculate_costs(
                        trade_df, pd.Series([exit_price], index=[symbol])
                    )[symbol]

                    net_proceeds = proceeds - costs
                    cash += net_proceeds

                    # Log trade
                    entry_price_avg = (
                        trade_log[-1]["price"] if trade_log and trade_log[-1]["symbol"] == symbol else exit_price
                    )
                    pnl = (execution_price - entry_price_avg) * shares_to_sell - costs

                    trade_log.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "action": "exit",
                            "shares": shares_to_sell,
                            "price": execution_price,
                            "notional": proceeds,
                            "costs": costs,
                            "pnl": pnl,
                        }
                    )

                    positions[symbol] = 0

            # Process entries (new positions)
            entry_signals = pd.DataFrame({"entry": entries})
            target_shares = self.sizing_func(
                entry_signals, equity, current_prices, max_positions=self.max_positions, **sizing_kwargs
            )

            for symbol in symbols:
                if target_shares[symbol] > 0 and positions[symbol] == 0:
                    shares_to_buy = target_shares[symbol]
                    entry_price = current_prices[symbol]

                    # Apply slippage
                    trade_df = pd.DataFrame({"shares": [shares_to_buy]}, index=[symbol])
                    execution_price = self.slippage_model.apply_slippage(
                        trade_df, pd.Series([entry_price], index=[symbol])
                    )[symbol]

                    # Calculate cost
                    cost = shares_to_buy * execution_price
                    costs = self.cost_model.calculate_costs(
                        trade_df, pd.Series([entry_price], index=[symbol])
                    )[symbol]

                    total_cost = cost + costs

                    # Check if we have enough cash
                    if total_cost <= cash:
                        cash -= total_cost
                        positions[symbol] = shares_to_buy

                        trade_log.append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "action": "entry",
                                "shares": shares_to_buy,
                                "price": execution_price,
                                "notional": cost,
                                "costs": costs,
                                "pnl": 0.0,  # Entry has no P&L yet
                            }
                        )

        # Final equity calculation
        final_prices = prices.iloc[-1]
        final_position_value = (positions * final_prices).sum()
        final_equity = cash + final_position_value
        equity_curve.append(final_equity)

        # Store results
        self.equity_curve = pd.Series(equity_curve, index=dates.tolist() + [dates[-1]])
        self.returns = self.equity_curve.pct_change().fillna(0)
        self.trades = pd.DataFrame(trade_log)
        self.positions = pd.DataFrame({"shares": positions})

        logger.info(f"Simulation complete: {len(trade_log)} trades, final equity ${final_equity:,.2f}")

    def print_results(self) -> None:
        """Print backtest results to console."""
        if self.metrics is None:
            logger.error("No results to print. Run backtest first.")
            return

        print(format_metrics(self.metrics))

    def save_results(self, output_dir: Path) -> None:
        """
        Save backtest results to files.

        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        # Save equity curve
        if self.equity_curve is not None:
            self.equity_curve.to_csv(output_dir / "equity_curve.csv", header=["equity"])

        # Save trades
        if self.trades is not None:
            self.trades.to_csv(output_dir / "trades.csv", index=False)

        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # This will be tested once we have strategies implemented
    logger.info("Backtest engine module loaded")
