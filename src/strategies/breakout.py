"""
Breakout Strategy (Donchian Channel).
Enters on breakout above N-period high, exits on breakout below N-period low.
Optional ATR-based stops.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.strategies.base import Strategy, atr, breakout_high, breakout_low


class BreakoutStrategy(Strategy):
    """
    Donchian channel breakout strategy.

    Entry: Close breaks above N-period high
    Exit: Close breaks below N-period low
    Optional: ATR-based trailing stop
    """

    def __init__(
        self, lookback: int = 20, atr_multiplier: float = 2.0, use_atr_stop: bool = False
    ) -> None:
        """
        Initialize breakout strategy.

        Args:
            lookback: Period for Donchian channel
            atr_multiplier: Multiplier for ATR-based stop
            use_atr_stop: Whether to use ATR stop (in addition to low breakout)
        """
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.use_atr_stop = use_atr_stop

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate breakout signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with 'entry', 'exit', and optional 'stop_loss' columns
        """
        close = df["close"] if "close" in df.columns else df["Close"]

        # Entry: breakout above N-period high
        entry = breakout_high(close, self.lookback)

        # Exit: breakout below N-period low
        exit_signal = breakout_low(close, self.lookback)

        signals = pd.DataFrame(
            {"entry": entry.fillna(False), "exit": exit_signal.fillna(False)}, index=df.index
        )

        # Optional: ATR-based stop loss
        if self.use_atr_stop:
            atr_value = atr(df, period=14)
            # Stop loss = entry price - (ATR * multiplier)
            # Store ATR value for use in backtest engine
            signals["atr_stop_distance"] = atr_value * self.atr_multiplier

        return signals


class BreakoutFromYAML(Strategy):
    """Breakout strategy driven by YAML parameters."""

    def __init__(self, params: dict[str, Any]) -> None:
        """
        Initialize from YAML parameters.

        Args:
            params: Dict with 'lookback', 'atr_multiplier', etc.
        """
        self.lookback = params.get("lookback", 20)
        self.atr_multiplier = params.get("atr_multiplier", 2.0)
        self.use_atr_stop = params.get("use_atr_stop", False)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout signals from YAML params."""
        strategy = BreakoutStrategy(
            lookback=self.lookback,
            atr_multiplier=self.atr_multiplier,
            use_atr_stop=self.use_atr_stop,
        )
        return strategy.generate_signals(df)


if __name__ == "__main__":
    # Test breakout strategy
    import numpy as np

    dates = pd.date_range("2020-01-01", periods=300, freq="D")

    # Create trending price data
    trend = np.linspace(100, 150, 300)
    noise = np.random.randn(300) * 3
    close_prices = trend + noise

    df = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices + 2,
            "low": close_prices - 2,
            "close": close_prices,
            "volume": 1_000_000,
        },
        index=dates,
    )

    strategy = BreakoutStrategy(lookback=55, atr_multiplier=2.0, use_atr_stop=True)
    signals = strategy.generate_signals(df)

    print("Breakout Signals:")
    print(signals[signals["entry"] | signals["exit"]].head(10))
    print(f"\nTotal entries: {signals['entry'].sum()}")
    print(f"Total exits: {signals['exit'].sum()}")
