"""
Abstract base class for data sources.
All data providers (yfinance, IBKR, etc.) implement this interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataSource(ABC):
    """Abstract interface for market data providers."""

    @abstractmethod
    def get_prices(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical price data for a list of symbols.

        Args:
            symbols: List of ticker symbols
            start: Start date (YYYY-MM-DD or datetime)
            end: End date (YYYY-MM-DD or datetime)
            interval: Data interval ('1d', '1h', '15m', etc.)

        Returns:
            Dict mapping symbol -> DataFrame with columns [open, high, low, close, volume]
            Index is datetime
        """
        pass

    @abstractmethod
    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        Get latest prices for a list of symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol -> latest price
        """
        pass

    def get_dividends(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> dict[str, pd.DataFrame]:
        """
        Get dividend data for a list of symbols (optional).

        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date

        Returns:
            Dict mapping symbol -> DataFrame with dividend data
        """
        # Default implementation returns empty data
        return {symbol: pd.DataFrame() for symbol in symbols}

    def get_contract_details(self, symbols: list[str]) -> dict[str, dict]:
        """
        Get contract details (exchange, currency, tick size, etc.) (optional).

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol -> contract details dict
        """
        # Default implementation returns minimal details
        return {
            symbol: {"exchange": "SMART", "currency": "USD", "sec_type": "STK"}
            for symbol in symbols
        }
