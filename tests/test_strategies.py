"""Test strategy implementations."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.breakout import BreakoutStrategy
from src.strategies.sma_cross import SMACrossStrategy


def create_test_dataframe(periods: int = 300) -> pd.DataFrame:
    """Create a test OHLCV dataframe."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    close = 100 + np.random.randn(periods).cumsum()

    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
        },
        index=dates,
    )


def test_sma_cross_strategy() -> None:
    """Test SMA crossover strategy."""
    df = create_test_dataframe()

    strategy = SMACrossStrategy(fast_period=20, slow_period=50)
    signals = strategy.generate_signals(df)

    assert "entry" in signals.columns
    assert "exit" in signals.columns
    assert len(signals) == len(df)
    assert signals["entry"].dtype == bool
    assert signals["exit"].dtype == bool


def test_breakout_strategy() -> None:
    """Test breakout strategy."""
    # Create trending data
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    close = np.linspace(100, 150, 300)

    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": 1_000_000,
        },
        index=dates,
    )

    strategy = BreakoutStrategy(lookback=55)
    signals = strategy.generate_signals(df)

    assert "entry" in signals.columns
    assert "exit" in signals.columns
    assert len(signals) == len(df)


def test_sma_cross_generates_signals() -> None:
    """Test that SMA cross generates some signals."""
    df = create_test_dataframe(periods=500)

    strategy = SMACrossStrategy(fast_period=20, slow_period=50)
    signals = strategy.generate_signals(df)

    # Should generate at least a few signals with 500 days of data
    total_signals = signals["entry"].sum() + signals["exit"].sum()
    assert total_signals > 0, "Strategy should generate some signals"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
