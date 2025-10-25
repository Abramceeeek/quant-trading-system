"""
IBKR data source using ib_insync.
Handles historical bars with rate-limiting, retries, and caching.
"""

import logging
import time
from datetime import datetime

import pandas as pd
from ib_insync import IB, Stock, util

from src.core.config import config
from src.datasource.base import DataSource

logger = logging.getLogger(__name__)


class IBKRSource(DataSource):
    """IBKR data provider using ib_insync."""

    def __init__(self, connect: bool = True) -> None:
        """
        Initialize IBKR connection.

        Args:
            connect: Whether to connect immediately (default True)
        """
        self.ib = IB()
        self.connected = False
        self.rate_limit_delay = 0.5  # Seconds between requests

        if connect:
            self.connect()

    def connect(self) -> None:
        """Connect to IBKR Gateway/TWS."""
        if self.connected:
            return

        try:
            self.ib.connect(
                host=config.ibkr_host,
                port=config.ibkr_port,
                clientId=config.ibkr_client_id,
                timeout=20,
            )
            self.connected = True
            logger.info(
                f"Connected to IBKR at {config.ibkr_host}:{config.ibkr_port} "
                f"(clientId={config.ibkr_client_id})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

    def get_prices(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical price data from IBKR.

        Args:
            symbols: List of ticker symbols
            start: Start date (YYYY-MM-DD or datetime)
            end: End date (YYYY-MM-DD or datetime)
            interval: Data interval ('1d', '1h', '15m', etc.)
                     Maps to IBKR bar sizes: 1d->1 day, 1h->1 hour, 15m->15 mins

        Returns:
            Dict mapping symbol -> DataFrame with [open, high, low, close, volume]
        """
        if not self.connected:
            self.connect()

        end_dt = pd.to_datetime(end)
        duration = self._calculate_duration(start, end)
        bar_size = self._map_interval_to_barsize(interval)

        result = {}

        for symbol in symbols:
            try:
                contract = Stock(symbol, "SMART", "USD")

                # Request historical data with retries
                bars = self._request_historical_with_retry(
                    contract, end_dt.strftime("%Y%m%d %H:%M:%S"), duration, bar_size
                )

                if bars:
                    df = util.df(bars)
                    # Normalize columns
                    df = df.rename(columns={"date": "datetime"})
                    df = df.set_index("datetime")
                    df.columns = df.columns.str.lower()

                    # Filter by start date
                    start_dt = pd.to_datetime(start)
                    df = df[df.index >= start_dt]

                    result[symbol] = df
                    logger.info(f"Fetched {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")

        return result

    def _request_historical_with_retry(
        self, contract: Stock, end_datetime: str, duration: str, bar_size: str, retries: int = 3
    ) -> list:
        """
        Request historical data with exponential backoff on failure.

        Args:
            contract: IB contract
            end_datetime: End datetime string
            duration: Duration string
            bar_size: Bar size string
            retries: Number of retry attempts

        Returns:
            List of Bar objects
        """
        for attempt in range(retries):
            try:
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_datetime,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,  # Regular trading hours
                )
                return bars
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Historical data request failed (attempt {attempt+1}/{retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Historical data request failed after {retries} attempts: {e}")
                    raise

        return []

    def _calculate_duration(self, start: str | datetime, end: str | datetime) -> str:
        """
        Calculate IBKR duration string from start and end dates.

        Args:
            start: Start date
            end: End date

        Returns:
            Duration string (e.g., '365 D', '52 W', '5 Y')
        """
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        days = (end_dt - start_dt).days

        if days <= 60:
            return f"{days} D"
        elif days <= 365:
            weeks = days // 7
            return f"{weeks} W"
        else:
            years = days // 365
            return f"{years} Y"

    def _map_interval_to_barsize(self, interval: str) -> str:
        """
        Map pandas/yfinance-style interval to IBKR bar size.

        Args:
            interval: Interval string ('1d', '1h', '15m', etc.)

        Returns:
            IBKR bar size string ('1 day', '1 hour', '15 mins', etc.)
        """
        mapping = {
            "1d": "1 day",
            "1h": "1 hour",
            "30m": "30 mins",
            "15m": "15 mins",
            "5m": "5 mins",
            "1m": "1 min",
        }
        return mapping.get(interval, "1 day")

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        Get latest market prices from IBKR.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol -> latest price
        """
        if not self.connected:
            self.connect()

        result = {}

        for symbol in symbols:
            try:
                contract = Stock(symbol, "SMART", "USD")
                self.ib.qualifyContracts(contract)

                # Request market data snapshot
                ticker = self.ib.reqMktData(contract, snapshot=True)
                self.ib.sleep(2)  # Wait for data

                if ticker.last and ticker.last > 0:
                    result[symbol] = float(ticker.last)
                elif ticker.close and ticker.close > 0:
                    result[symbol] = float(ticker.close)
                else:
                    logger.warning(f"No valid price for {symbol}")

                self.ib.cancelMktData(contract)

            except Exception as e:
                logger.error(f"Failed to fetch latest price for {symbol}: {e}")

        return result

    def __enter__(self) -> "IBKRSource":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.disconnect()


if __name__ == "__main__":
    # Test IBKR connection and data fetch
    logging.basicConfig(level=logging.INFO)

    try:
        with IBKRSource() as source:
            # Test historical data
            data = source.get_prices(["AAPL", "MSFT"], "2024-10-01", "2024-10-25", "1d")
            for sym, df in data.items():
                print(f"\n{sym}:")
                print(df.head())

            # Test latest prices
            latest = source.get_latest_prices(["AAPL", "MSFT", "SPY"])
            print(f"\nLatest prices: {latest}")

    except Exception as e:
        print(f"IBKR test failed: {e}")
        print("Make sure TWS/Gateway is running and API is enabled")
