"""
Slippage models for realistic execution simulation.
Accounts for bid-ask spread, market impact, and price movement during execution.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SlippageModel:
    """Base slippage model."""

    def apply_slippage(
        self,
        trades: pd.DataFrame,
        prices: pd.Series,
        volumes: pd.Series | None = None,
    ) -> pd.Series:
        """
        Apply slippage to trade prices.

        Args:
            trades: DataFrame with 'shares' column (signed: + for buy, - for sell)
            prices: Series of mid prices at trade signal
            volumes: Optional series of daily volumes (for impact models)

        Returns:
            Series of execution prices (after slippage)
        """
        raise NotImplementedError


class FixedSlippage(SlippageModel):
    """Fixed percentage slippage (simplest model)."""

    def __init__(self, slippage_pct: float = 0.0005) -> None:
        """
        Initialize fixed slippage model.

        Args:
            slippage_pct: Slippage as decimal (0.0005 = 0.05% = 5 bps)
        """
        self.slippage_pct = slippage_pct

    def apply_slippage(
        self,
        trades: pd.DataFrame,
        prices: pd.Series,
        volumes: pd.Series | None = None,
    ) -> pd.Series:
        # Buy: pay more, Sell: receive less
        direction = np.sign(trades["shares"])  # +1 for buy, -1 for sell
        slippage = prices * self.slippage_pct * direction
        return prices + slippage


class VolumeSlippage(SlippageModel):
    """
    Volume-based slippage (market impact).
    Slippage increases with trade size relative to average daily volume.
    """

    def __init__(
        self,
        base_slippage_pct: float = 0.0005,
        impact_coeff: float = 0.1,
    ) -> None:
        """
        Initialize volume-based slippage model.

        Args:
            base_slippage_pct: Base slippage (like bid-ask spread)
            impact_coeff: Market impact coefficient
        """
        self.base_slippage_pct = base_slippage_pct
        self.impact_coeff = impact_coeff

    def apply_slippage(
        self,
        trades: pd.DataFrame,
        prices: pd.Series,
        volumes: pd.Series | None = None,
    ) -> pd.Series:
        if volumes is None:
            # Fallback to fixed slippage if no volume data
            logger.warning("No volume data provided, using base slippage only")
            return FixedSlippage(self.base_slippage_pct).apply_slippage(
                trades, prices, volumes
            )

        # Calculate participation rate (trade size / avg daily volume)
        trade_shares = trades["shares"].abs()
        participation = (trade_shares / volumes).fillna(0).clip(upper=0.5)

        # Impact: sqrt(participation) * coefficient
        # Square root to account for non-linear market impact
        impact = self.impact_coeff * np.sqrt(participation)

        # Total slippage = base + impact
        total_slippage_pct = self.base_slippage_pct + impact

        direction = np.sign(trades["shares"])
        slippage = prices * total_slippage_pct * direction

        return prices + slippage


class BidAskSlippage(SlippageModel):
    """
    Bid-ask spread slippage.
    Assumes trades cross the spread (buy at ask, sell at bid).
    """

    def __init__(self, half_spread_pct: float = 0.0005) -> None:
        """
        Initialize bid-ask slippage model.

        Args:
            half_spread_pct: Half spread as decimal (full spread = 2x this)
        """
        self.half_spread_pct = half_spread_pct

    def apply_slippage(
        self,
        trades: pd.DataFrame,
        prices: pd.Series,
        volumes: pd.Series | None = None,
    ) -> pd.Series:
        # Buy at ask (mid + half_spread), Sell at bid (mid - half_spread)
        direction = np.sign(trades["shares"])
        slippage = prices * self.half_spread_pct * direction
        return prices + slippage


def get_slippage_model(model_type: str = "fixed", **kwargs: float) -> SlippageModel:
    """
    Factory function to create slippage models.

    Args:
        model_type: 'fixed', 'volume', 'bidask'
        **kwargs: Model-specific parameters

    Returns:
        SlippageModel instance
    """
    if model_type == "fixed":
        return FixedSlippage(kwargs.get("slippage_pct", 0.0005))
    elif model_type == "volume":
        return VolumeSlippage(
            kwargs.get("base_slippage_pct", 0.0005),
            kwargs.get("impact_coeff", 0.1),
        )
    elif model_type == "bidask":
        return BidAskSlippage(kwargs.get("half_spread_pct", 0.0005))
    else:
        raise ValueError(f"Unknown slippage model: {model_type}")


if __name__ == "__main__":
    # Test slippage models
    trades = pd.DataFrame({"shares": [100, -50, 200, -150]})
    prices = pd.Series([150.0, 151.0, 149.0, 152.0])
    volumes = pd.Series([1_000_000, 500_000, 800_000, 1_200_000])

    print("Fixed slippage:")
    print(FixedSlippage(0.001).apply_slippage(trades, prices))

    print("\nVolume slippage:")
    print(VolumeSlippage().apply_slippage(trades, prices, volumes))

    print("\nBid-ask slippage:")
    print(BidAskSlippage(0.0005).apply_slippage(trades, prices))
