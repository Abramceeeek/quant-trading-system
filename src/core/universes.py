"""
Universe definitions and management.
Provides predefined universes (S&P 500, liquid stocks, demo), liquidity filters,
and reconstitution logic.
"""

import logging
from pathlib import Path

import pandas as pd

from src.core.config import config

logger = logging.getLogger(__name__)

# Demo universe for quick testing
DEMO_UNIVERSE = ["AAPL", "MSFT", "SPY", "QQQ", "IWM"]


def get_universe(name: str) -> list[str]:
    """
    Get a universe by name.

    Args:
        name: Universe name (us_equities_core, us_equities_liquid, demo, custom_universe)

    Returns:
        List of ticker symbols
    """
    if name == "demo":
        return DEMO_UNIVERSE
    elif name == "us_equities_core":
        return get_us_equities_core()
    elif name == "us_equities_liquid":
        return get_us_equities_liquid()
    elif name == "custom_universe":
        return get_custom_universe()
    else:
        raise ValueError(f"Unknown universe: {name}")


def get_us_equities_core() -> list[str]:
    """
    Get S&P 500 universe.
    Reads from data/universes/sp500.csv if available, otherwise returns demo list.

    Returns:
        List of S&P 500 ticker symbols
    """
    sp500_file = config.universes_dir / "sp500.csv"

    if sp500_file.exists():
        try:
            df = pd.read_csv(sp500_file)
            # Assume CSV has a 'ticker' or 'symbol' column
            if "ticker" in df.columns:
                tickers = df["ticker"].dropna().str.strip().tolist()
            elif "symbol" in df.columns:
                tickers = df["symbol"].dropna().str.strip().tolist()
            else:
                # Fallback: use first column
                tickers = df.iloc[:, 0].dropna().str.strip().tolist()

            logger.info(f"Loaded {len(tickers)} tickers from {sp500_file}")
            return tickers
        except Exception as e:
            logger.warning(f"Failed to load {sp500_file}: {e}. Using demo universe.")
            return DEMO_UNIVERSE
    else:
        logger.warning(
            f"{sp500_file} not found. Using demo universe. "
            "Download S&P 500 constituents to data/universes/sp500.csv"
        )
        return DEMO_UNIVERSE


def get_us_equities_liquid() -> list[str]:
    """
    Get liquid US equities (filtered S&P 500).
    In a full implementation, this would filter by:
    - Avg daily volume > 1M shares
    - Price > $10
    - Market cap > $1B

    For now, returns core universe (filters can be applied in backtest preprocessing).

    Returns:
        List of liquid ticker symbols
    """
    # Placeholder: In practice, apply filters to historical data
    # For now, just return the core universe
    core = get_us_equities_core()
    logger.info(f"Liquid universe: {len(core)} tickers (no filters applied yet)")
    return core


def get_custom_universe() -> list[str]:
    """
    Get a custom universe.
    Users can modify this function or add additional custom universes.

    Returns:
        List of custom ticker symbols
    """
    return ["AAPL", "GOOGL", "AMZN", "TSLA", "NVDA", "MSFT", "META", "NFLX"]


def get_recon_dates(start: str, end: str, freq: str = "MS") -> list[str]:
    """
    Get reconstitution dates for universe rebalancing.

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        freq: Pandas frequency string ('MS' = month start, 'QS' = quarter start, etc.)

    Returns:
        List of rebalance dates as YYYY-MM-DD strings
    """
    dates = pd.date_range(start=start, end=end, freq=freq)
    return [d.strftime("%Y-%m-%d") for d in dates]


def apply_liquidity_filter(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    min_price: float = 10.0,
    min_volume: float = 1_000_000,
    lookback_days: int = 20,
) -> list[str]:
    """
    Apply liquidity filters to a universe.

    Args:
        tickers: List of tickers to filter
        price_data: Dict of {ticker: DataFrame with OHLCV}
        min_price: Minimum average price over lookback period
        min_volume: Minimum average volume over lookback period
        lookback_days: Number of days to average over

    Returns:
        Filtered list of tickers
    """
    filtered = []

    for ticker in tickers:
        if ticker not in price_data:
            continue

        df = price_data[ticker]
        if len(df) < lookback_days:
            continue

        # Check average price and volume over lookback period
        recent = df.tail(lookback_days)
        avg_price = recent["close"].mean()
        avg_volume = recent["volume"].mean()

        if avg_price >= min_price and avg_volume >= min_volume:
            filtered.append(ticker)

    logger.info(
        f"Liquidity filter: {len(filtered)}/{len(tickers)} tickers passed "
        f"(price >= ${min_price}, volume >= {min_volume:,.0f})"
    )
    return filtered


def create_sp500_csv_example() -> None:
    """
    Helper function to create an example sp500.csv file.
    Run this once to bootstrap the universe file.
    """
    example_tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "BRK.B",
        "UNH",
        "JNJ",
        "V",
        "XOM",
        "WMT",
        "JPM",
        "PG",
        "MA",
        "HD",
        "CVX",
        "LLY",
        "ABBV",
        # Add more as needed, or download full S&P 500 list
    ]

    output_file = config.universes_dir / "sp500.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"ticker": example_tickers})
    df.to_csv(output_file, index=False)
    logger.info(f"Created example universe file: {output_file}")


if __name__ == "__main__":
    # Generate example sp500.csv
    import sys

    logging.basicConfig(level=logging.INFO)

    if "--create-csv" in sys.argv:
        create_sp500_csv_example()
        print("Created data/universes/sp500.csv with example tickers")
    else:
        # Test universes
        print("Demo universe:", get_universe("demo"))
        print("US Equities Core:", get_universe("us_equities_core"))
        print("Reconstitution dates (2023):", get_recon_dates("2023-01-01", "2023-12-31", "MS"))
