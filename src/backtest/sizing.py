"""
Position sizing logic.
Supports equal weight, volatility targeting, Kelly criterion, fixed risk, etc.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight_sizing(
    signals: pd.DataFrame,
    equity: float,
    prices: pd.Series,
    max_positions: int = 20,
) -> pd.Series:
    """
    Equal weight position sizing.

    Args:
        signals: DataFrame with 'entry' column (boolean or 0/1)
        equity: Current portfolio equity
        prices: Current prices for each symbol
        max_positions: Maximum number of positions

    Returns:
        Series of target shares for each symbol
    """
    active_signals = signals["entry"].astype(bool)
    n_signals = active_signals.sum()

    if n_signals == 0:
        return pd.Series(0, index=signals.index)

    # Limit to max_positions
    n_positions = min(n_signals, max_positions)

    # Equal allocation per position
    allocation_per_position = equity / n_positions

    # Calculate shares
    shares = pd.Series(0, index=signals.index)
    shares[active_signals] = (allocation_per_position / prices[active_signals]).fillna(0)

    # If more signals than max_positions, keep only the first N
    if n_signals > max_positions:
        top_signals = active_signals[active_signals].index[:max_positions]
        mask = pd.Series(False, index=signals.index)
        mask[top_signals] = True
        shares[~mask] = 0

    return shares.astype(int)


def vol_target_sizing(
    signals: pd.DataFrame,
    equity: float,
    prices: pd.Series,
    returns: pd.DataFrame,
    target_vol: float = 0.15,
    lookback: int = 60,
    max_positions: int = 20,
) -> pd.Series:
    """
    Volatility-targeted position sizing.
    Scales positions inversely to realized volatility.

    Args:
        signals: DataFrame with 'entry' column
        equity: Current portfolio equity
        prices: Current prices
        returns: Historical returns DataFrame (for vol calculation)
        target_vol: Target portfolio volatility (annualized)
        lookback: Lookback period for volatility calculation
        max_positions: Maximum number of positions

    Returns:
        Series of target shares for each symbol
    """
    active_signals = signals["entry"].astype(bool)
    n_signals = active_signals.sum()

    if n_signals == 0:
        return pd.Series(0, index=signals.index)

    n_positions = min(n_signals, max_positions)

    # Calculate realized volatility for each symbol
    recent_returns = returns.tail(lookback)
    realized_vol = recent_returns.std() * np.sqrt(252)  # Annualized

    # Inverse volatility weighting
    inv_vol = 1 / realized_vol.replace(0, np.nan)
    inv_vol = inv_vol.fillna(0)

    # Normalize weights
    active_inv_vol = inv_vol[active_signals]
    if active_inv_vol.sum() > 0:
        weights = active_inv_vol / active_inv_vol.sum()
    else:
        # Fallback to equal weight
        weights = pd.Series(1.0 / n_positions, index=active_inv_vol.index)

    # Scale by target vol
    # target_vol / current_vol determines leverage factor
    # For simplicity, assume current_vol ≈ average realized_vol
    avg_realized_vol = realized_vol[active_signals].mean()
    if avg_realized_vol > 0:
        vol_scalar = target_vol / avg_realized_vol
    else:
        vol_scalar = 1.0

    # Calculate dollar allocations
    allocations = weights * equity * vol_scalar

    # Convert to shares
    shares = pd.Series(0, index=signals.index)
    shares[active_signals] = (allocations / prices[active_signals]).fillna(0)

    # Apply max_positions limit (keep top weighted)
    if n_signals > max_positions:
        top_signals = weights.nlargest(max_positions).index
        mask = pd.Series(False, index=signals.index)
        mask[top_signals] = True
        shares[~mask] = 0

    return shares.astype(int)


def fixed_risk_sizing(
    signals: pd.DataFrame,
    equity: float,
    prices: pd.Series,
    stop_loss_pct: float = 0.05,
    risk_per_trade_pct: float = 0.01,
) -> pd.Series:
    """
    Fixed risk per trade position sizing.
    Size each position so that hitting the stop loss = risk_per_trade_pct of equity.

    Args:
        signals: DataFrame with 'entry' column
        equity: Current portfolio equity
        prices: Current prices
        stop_loss_pct: Stop loss distance as % of entry price
        risk_per_trade_pct: Risk per trade as % of equity

    Returns:
        Series of target shares for each symbol
    """
    active_signals = signals["entry"].astype(bool)

    if not active_signals.any():
        return pd.Series(0, index=signals.index)

    # Risk per trade in dollars
    risk_per_trade = equity * risk_per_trade_pct

    # Risk per share = entry_price * stop_loss_pct
    risk_per_share = prices * stop_loss_pct

    # Shares = risk_per_trade / risk_per_share
    shares = pd.Series(0, index=signals.index)
    shares[active_signals] = (risk_per_trade / risk_per_share[active_signals]).fillna(0)

    return shares.astype(int)


def kelly_criterion_sizing(
    signals: pd.DataFrame,
    equity: float,
    prices: pd.Series,
    win_rate: float = 0.55,
    avg_win: float = 0.015,
    avg_loss: float = 0.010,
    max_positions: int = 20,
    kelly_fraction: float = 0.25,
) -> pd.Series:
    """
    Kelly Criterion position sizing (simplified).

    Kelly % = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

    Args:
        signals: DataFrame with 'entry' column
        equity: Current portfolio equity
        prices: Current prices
        win_rate: Historical win rate (0-1)
        avg_win: Average win as fraction
        avg_loss: Average loss as fraction
        max_positions: Maximum positions
        kelly_fraction: Fractional Kelly (0.25 = quarter Kelly for safety)

    Returns:
        Series of target shares for each symbol
    """
    active_signals = signals["entry"].astype(bool)
    n_signals = active_signals.sum()

    if n_signals == 0:
        return pd.Series(0, index=signals.index)

    # Kelly formula
    kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    kelly_pct = max(kelly_pct, 0)  # No shorting if negative

    # Apply fractional Kelly for safety
    position_size_pct = kelly_pct * kelly_fraction

    # Divide across positions
    n_positions = min(n_signals, max_positions)
    allocation_per_position = (equity * position_size_pct) / n_positions

    # Calculate shares
    shares = pd.Series(0, index=signals.index)
    shares[active_signals] = (allocation_per_position / prices[active_signals]).fillna(0)

    # Limit to max_positions
    if n_signals > max_positions:
        top_signals = active_signals[active_signals].index[:max_positions]
        mask = pd.Series(False, index=signals.index)
        mask[top_signals] = True
        shares[~mask] = 0

    return shares.astype(int)


def get_sizing_method(method: str) -> callable:
    """
    Factory function for sizing methods.

    Args:
        method: 'equal_weight', 'vol_target', 'fixed_risk', 'kelly'

    Returns:
        Sizing function
    """
    methods = {
        "equal_weight": equal_weight_sizing,
        "vol_target": vol_target_sizing,
        "fixed_risk": fixed_risk_sizing,
        "kelly": kelly_criterion_sizing,
    }

    if method not in methods:
        raise ValueError(f"Unknown sizing method: {method}. Choose from {list(methods.keys())}")

    return methods[method]


if __name__ == "__main__":
    # Test sizing methods
    signals = pd.DataFrame(
        {"entry": [True, True, False, True, True]},
        index=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    )
    prices = pd.Series([180.0, 350.0, 140.0, 170.0, 250.0], index=signals.index)
    equity = 100_000

    print("Equal weight:")
    print(equal_weight_sizing(signals, equity, prices, max_positions=3))

    print("\nFixed risk:")
    print(fixed_risk_sizing(signals, equity, prices, stop_loss_pct=0.05, risk_per_trade_pct=0.01))

    print("\nKelly:")
    print(
        kelly_criterion_sizing(
            signals, equity, prices, win_rate=0.55, avg_win=0.02, avg_loss=0.01
        )
    )
