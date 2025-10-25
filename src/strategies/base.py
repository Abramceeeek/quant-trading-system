"""
Strategy base class and rule parsing utilities.
Provides a simple DSL for YAML-based strategy authoring.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from price data.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with boolean columns 'entry' and 'exit' (same index as input)
        """
        pass


# Technical indicator helpers


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# Signal helpers


def crosses_above(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses above series2."""
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crosses_below(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses below series2."""
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def breakout_high(series: pd.Series, period: int) -> pd.Series:
    """Detect breakout above N-period high."""
    rolling_high = series.shift(1).rolling(window=period, min_periods=period).max()
    return series > rolling_high


def breakout_low(series: pd.Series, period: int) -> pd.Series:
    """Detect breakout below N-period low."""
    rolling_low = series.shift(1).rolling(window=period, min_periods=period).min()
    return series < rolling_low


# Simple rule parser for YAML


def parse_rule(series: pd.Series, rule: str) -> pd.Series:
    """
    Parse a simple rule string and return boolean signal series.

    Supported rules:
    - "SMA(N) crosses above SMA(M)"
    - "SMA(N) crosses below SMA(M)"
    - "close crosses above SMA(N)"
    - "close crosses below SMA(N)"

    Args:
        series: Price series (typically close)
        rule: Rule string

    Returns:
        Boolean series indicating signal
    """
    rule_clean = rule.replace(" ", "").lower()

    # SMA cross patterns
    pattern1 = r"sma\((\d+)\)crosses(above|below)sma\((\d+)\)"
    match1 = re.match(pattern1, rule_clean)

    if match1:
        period1 = int(match1.group(1))
        direction = match1.group(2)
        period2 = int(match1.group(3))

        sma1 = sma(series, period1)
        sma2 = sma(series, period2)

        if direction == "above":
            return crosses_above(sma1, sma2).fillna(False)
        else:
            return crosses_below(sma1, sma2).fillna(False)

    # Close vs SMA patterns
    pattern2 = r"closecrosses(above|below)sma\((\d+)\)"
    match2 = re.match(pattern2, rule_clean)

    if match2:
        direction = match2.group(1)
        period = int(match2.group(2))

        sma_series = sma(series, period)

        if direction == "above":
            return crosses_above(series, sma_series).fillna(False)
        else:
            return crosses_below(series, sma_series).fillna(False)

    raise ValueError(f"Unsupported rule: {rule}")
