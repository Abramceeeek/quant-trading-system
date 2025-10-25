"""
SMA Crossover Strategy.
Generates signals based on simple or exponential moving average crossovers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.strategies.base import Strategy, parse_rule, sma


class SMACrossStrategy(Strategy):
    """
    SMA crossover strategy.

    Entry: Fast SMA crosses above slow SMA
    Exit: Fast SMA crosses below slow SMA
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200) -> None:
        """
        Initialize SMA crossover strategy.

        Args:
            fast_period: Fast SMA period
            slow_period: Slow SMA period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate SMA crossover signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with 'entry' and 'exit' boolean columns
        """
        close = df["close"] if "close" in df.columns else df["Close"]

        fast_sma = sma(close, self.fast_period)
        slow_sma = sma(close, self.slow_period)

        # Entry: fast crosses above slow
        entry = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))

        # Exit: fast crosses below slow
        exit_signal = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))

        return pd.DataFrame(
            {"entry": entry.fillna(False), "exit": exit_signal.fillna(False)}, index=df.index
        )


class SMACrossFromYAML(Strategy):
    """
    SMA crossover strategy driven by YAML rules.
    Parses entry/exit rules from YAML parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        """
        Initialize from YAML parameters.

        Args:
            params: Dict with 'entry' and 'exit' rule strings
        """
        self.params = params

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals from YAML rules.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with 'entry' and 'exit' boolean columns
        """
        close = df["close"] if "close" in df.columns else df["Close"]

        entry_rule = self.params.get("entry_rule", "SMA(50) crosses above SMA(200)")
        exit_rule = self.params.get("exit_rule", "SMA(50) crosses below SMA(200)")

        entry = parse_rule(close, entry_rule)
        exit_signal = parse_rule(close, exit_rule)

        return pd.DataFrame(
            {"entry": entry.fillna(False), "exit": exit_signal.fillna(False)}, index=df.index
        )


if __name__ == "__main__":
    # Test SMA cross strategy
    import numpy as np

    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    close_prices = 100 + np.cumsum(np.random.randn(300) * 2)

    df = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices + 1,
            "low": close_prices - 1,
            "close": close_prices,
            "volume": 1_000_000,
        },
        index=dates,
    )

    strategy = SMACrossStrategy(fast_period=20, slow_period=50)
    signals = strategy.generate_signals(df)

    print("SMA Cross Signals:")
    print(signals[signals["entry"] | signals["exit"]].head(10))
    print(f"\nTotal entries: {signals['entry'].sum()}")
    print(f"Total exits: {signals['exit'].sum()}")
