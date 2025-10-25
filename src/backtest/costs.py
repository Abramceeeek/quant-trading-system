"""
Transaction cost models (commissions, fees).
Supports fixed per-share, percentage-based, and tiered commission structures.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class CostModel:
    """Base transaction cost model."""

    def calculate_costs(
        self, trades: pd.DataFrame, prices: pd.Series
    ) -> pd.Series:
        """
        Calculate transaction costs for a series of trades.

        Args:
            trades: DataFrame with 'shares' column (signed: + for buy, - for sell)
            prices: Series of prices at trade execution

        Returns:
            Series of costs (always positive)
        """
        raise NotImplementedError


class FixedPerShareCost(CostModel):
    """Fixed cost per share (e.g., $0.005/share)."""

    def __init__(self, cost_per_share: float = 0.005) -> None:
        self.cost_per_share = cost_per_share

    def calculate_costs(
        self, trades: pd.DataFrame, prices: pd.Series
    ) -> pd.Series:
        shares = trades["shares"].abs()
        return shares * self.cost_per_share


class PercentageCost(CostModel):
    """Percentage-based commission (e.g., 0.1% = 10 bps)."""

    def __init__(self, commission_pct: float = 0.001) -> None:
        """
        Initialize percentage cost model.

        Args:
            commission_pct: Commission as decimal (0.001 = 0.1% = 10 bps)
        """
        self.commission_pct = commission_pct

    def calculate_costs(
        self, trades: pd.DataFrame, prices: pd.Series
    ) -> pd.Series:
        notional = (trades["shares"].abs() * prices).fillna(0)
        return notional * self.commission_pct


class TieredCost(CostModel):
    """
    Tiered commission structure (e.g., lower rate for higher volumes).

    Example:
        - 0-10k shares: $0.005/share
        - 10k-100k shares: $0.003/share
        - 100k+ shares: $0.001/share
    """

    def __init__(
        self,
        tiers: list[tuple[float, float]] = [
            (10_000, 0.005),
            (100_000, 0.003),
            (float("inf"), 0.001),
        ],
    ) -> None:
        """
        Initialize tiered cost model.

        Args:
            tiers: List of (max_shares, cost_per_share) tuples
        """
        self.tiers = sorted(tiers, key=lambda x: x[0])

    def calculate_costs(
        self, trades: pd.DataFrame, prices: pd.Series
    ) -> pd.Series:
        shares = trades["shares"].abs()
        costs = pd.Series(0.0, index=trades.index)

        for max_shares, cost_per_share in self.tiers:
            applicable = shares <= max_shares
            costs[applicable] = shares[applicable] * cost_per_share
            if applicable.all():
                break

        return costs


class IBKRCost(CostModel):
    """
    IBKR Tiered pricing (simplified).

    Actual IBKR pricing:
    - Stocks: $0.0035/share, min $0.35, max 1% of trade value
    - This is a simplified approximation.
    """

    def __init__(
        self,
        per_share: float = 0.0035,
        min_per_order: float = 0.35,
        max_pct: float = 0.01,
    ) -> None:
        self.per_share = per_share
        self.min_per_order = min_per_order
        self.max_pct = max_pct

    def calculate_costs(
        self, trades: pd.DataFrame, prices: pd.Series
    ) -> pd.Series:
        shares = trades["shares"].abs()
        notional = (shares * prices).fillna(0)

        # Base cost
        costs = shares * self.per_share

        # Apply min per order
        costs = costs.clip(lower=self.min_per_order)

        # Apply max % of trade value
        max_cost = notional * self.max_pct
        costs = costs.clip(upper=max_cost)

        return costs


def get_cost_model(model_type: str = "percentage", **kwargs: float) -> CostModel:
    """
    Factory function to create cost models.

    Args:
        model_type: 'fixed', 'percentage', 'tiered', 'ibkr'
        **kwargs: Model-specific parameters

    Returns:
        CostModel instance
    """
    if model_type == "fixed":
        return FixedPerShareCost(kwargs.get("cost_per_share", 0.005))
    elif model_type == "percentage":
        return PercentageCost(kwargs.get("commission_pct", 0.001))
    elif model_type == "tiered":
        return TieredCost()
    elif model_type == "ibkr":
        return IBKRCost()
    else:
        raise ValueError(f"Unknown cost model: {model_type}")


if __name__ == "__main__":
    # Test cost models
    trades = pd.DataFrame({"shares": [100, -50, 200, -150]})
    prices = pd.Series([150.0, 151.0, 149.0, 152.0])

    print("Fixed cost:", FixedPerShareCost().calculate_costs(trades, prices))
    print("Percentage cost:", PercentageCost(0.001).calculate_costs(trades, prices))
    print("IBKR cost:", IBKRCost().calculate_costs(trades, prices))
