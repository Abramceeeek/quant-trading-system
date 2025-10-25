"""Risk guards for live trading."""

import pandas as pd


def check_max_drawdown(equity: pd.Series, threshold: float = 0.20) -> bool:
    """Check if drawdown exceeds threshold."""
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    return abs(drawdown.min()) > threshold
