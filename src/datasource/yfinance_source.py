"""
YFinance data source with parquet caching.
Supports daily and intraday data with automatic cache management.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.core.config import config
from src.datasource.base import DataSource

logger = logging.getLogger(__name__)


class YFinanceSource(DataSource):
    """YFinance data provider with parquet caching."""

    def __init__(self) -> None:
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = config.cache_enabled
        self.cache_expiry_days = config.cache_expiry_days

    def get_prices(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical price data with caching.

        Args:
            symbols: List of ticker symbols
            start: Start date (YYYY-MM-DD or datetime)
            end: End date (YYYY-MM-DD or datetime)
            interval: Data interval ('1d', '1h', '15m', etc.)

        Returns:
            Dict mapping symbol -> DataFrame with [open, high, low, close, volume]
        """
        start_str = start if isinstance(start, str) else start.strftime("%Y-%m-%d")
        end_str = end if isinstance(end, str) else end.strftime("%Y-%m-%d")

        result = {}

        for symbol in symbols:
            try:
                df = self._get_cached_or_download(symbol, start_str, end_str, interval)
                if df is not None and not df.empty:
                    # Normalize column names to lowercase
                    df.columns = df.columns.str.lower()
                    result[symbol] = df
                else:
                    logger.warning(f"No data for {symbol} in range {start_str} to {end_str}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")

        return result

    def _get_cached_or_download(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame | None:
        """
        Check cache, download if missing or expired.

        Args:
            symbol: Ticker symbol
            start: Start date string
            end: End date string
            interval: Data interval

        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start}_{end}.parquet"

        # Check cache
        if self.cache_enabled and cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age.days < self.cache_expiry_days:
                logger.debug(f"Loading {symbol} from cache: {cache_file}")
                return pd.read_parquet(cache_file)
            else:
                logger.debug(f"Cache expired for {symbol}, re-downloading")

        # Download from yfinance
        logger.info(f"Downloading {symbol} data from yfinance ({start} to {end}, {interval})")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Cache to parquet
            if self.cache_enabled:
                df.to_parquet(cache_file)
                logger.debug(f"Cached {symbol} to {cache_file}")

            return df

        except Exception as e:
            logger.error(f"yfinance download failed for {symbol}: {e}")
            return None

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        Get latest prices (current close).

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol -> latest price
        """
        result = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Get last 2 days to ensure we have recent data
                df = ticker.history(period="2d", interval="1d")
                if not df.empty:
                    result[symbol] = float(df["Close"].iloc[-1])
                else:
                    logger.warning(f"No recent data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch latest price for {symbol}: {e}")

        return result

    def get_dividends(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> dict[str, pd.DataFrame]:
        """
        Get dividend data from yfinance.

        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date

        Returns:
            Dict mapping symbol -> DataFrame with dividend data
        """
        result = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                divs = ticker.dividends
                if not divs.empty:
                    # Filter by date range
                    start_dt = pd.to_datetime(start)
                    end_dt = pd.to_datetime(end)
                    divs = divs[(divs.index >= start_dt) & (divs.index <= end_dt)]
                    result[symbol] = divs.to_frame(name="dividend")
                else:
                    result[symbol] = pd.DataFrame()
            except Exception as e:
                logger.error(f"Failed to fetch dividends for {symbol}: {e}")
                result[symbol] = pd.DataFrame()

        return result

    def clear_cache(self, symbol: str | None = None) -> None:
        """
        Clear cache for a specific symbol or all symbols.

        Args:
            symbol: Ticker symbol to clear (None = clear all)
        """
        if symbol:
            pattern = f"{symbol}_*.parquet"
        else:
            pattern = "*.parquet"

        count = 0
        for f in self.cache_dir.glob(pattern):
            f.unlink()
            count += 1

        logger.info(f"Cleared {count} cache file(s)")


if __name__ == "__main__":
    # Test the yfinance source
    logging.basicConfig(level=logging.INFO)

    source = YFinanceSource()

    # Test download
    data = source.get_prices(["AAPL", "MSFT"], "2024-01-01", "2024-12-31", "1d")
    for sym, df in data.items():
        print(f"\n{sym}:")
        print(df.head())
        print(f"Shape: {df.shape}")

    # Test latest prices
    latest = source.get_latest_prices(["AAPL", "MSFT", "SPY"])
    print(f"\nLatest prices: {latest}")
