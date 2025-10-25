"""
Performance metrics for backtesting.
Calculates CAGR, Sharpe, Sortino, Calmar, MaxDD, turnover, hit rate, etc.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cagr(equity_curve: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        equity_curve: Series of portfolio equity values

    Returns:
        CAGR as decimal (0.15 = 15% annual return)
    """
    if len(equity_curve) < 2:
        return 0.0

    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]

    if start_value <= 0 or end_value <= 0:
        return 0.0

    years = len(equity_curve) / 252.0  # Assuming daily data
    return (end_value / start_value) ** (1 / years) - 1


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Sharpe ratio (annualized)
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / 252.0  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino Ratio (downside deviation instead of total volatility).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sortino ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / 252.0
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        equity_curve: Series of portfolio equity values

    Returns:
        Max drawdown as positive decimal (0.20 = 20% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0

    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return abs(drawdown.min())


def calmar_ratio(equity_curve: pd.Series) -> float:
    """
    Calculate Calmar Ratio (CAGR / Max Drawdown).

    Args:
        equity_curve: Series of portfolio equity values

    Returns:
        Calmar ratio
    """
    cagr_val = cagr(equity_curve)
    max_dd = max_drawdown(equity_curve)

    if max_dd == 0:
        return 0.0

    return cagr_val / max_dd


def annual_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Series of returns

    Returns:
        Annual volatility as decimal
    """
    if len(returns) < 2:
        return 0.0

    return returns.std() * np.sqrt(252)


def turnover(trades: pd.DataFrame, equity_curve: pd.Series) -> float:
    """
    Calculate portfolio turnover (annual).

    Args:
        trades: DataFrame with 'notional' column (absolute trade values)
        equity_curve: Series of portfolio equity values

    Returns:
        Annual turnover as multiple of portfolio size
    """
    if len(trades) == 0 or len(equity_curve) == 0:
        return 0.0

    total_traded = trades["notional"].abs().sum()
    avg_equity = equity_curve.mean()
    days = len(equity_curve)

    if avg_equity == 0 or days == 0:
        return 0.0

    # Annualize
    annual_turnover = (total_traded / avg_equity) * (252.0 / days)
    return annual_turnover


def win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate win rate (% of profitable trades).

    Args:
        trades: DataFrame with 'pnl' column

    Returns:
        Win rate as decimal (0.55 = 55%)
    """
    if len(trades) == 0:
        return 0.0

    wins = (trades["pnl"] > 0).sum()
    return wins / len(trades)


def profit_factor(trades: pd.DataFrame) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        trades: DataFrame with 'pnl' column

    Returns:
        Profit factor (>1 is profitable)
    """
    if len(trades) == 0:
        return 0.0

    gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def avg_trade(trades: pd.DataFrame) -> float:
    """
    Calculate average trade P&L.

    Args:
        trades: DataFrame with 'pnl' column

    Returns:
        Average P&L per trade
    """
    if len(trades) == 0:
        return 0.0

    return trades["pnl"].mean()


def calculate_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate all performance metrics.

    Args:
        equity_curve: Series of portfolio equity values
        returns: Series of daily returns
        trades: DataFrame with trade details (pnl, notional, etc.)

    Returns:
        Dict of metric name -> value
    """
    metrics = {
        "cagr": cagr(equity_curve),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "annual_vol": annual_volatility(returns),
        "turnover": turnover(trades, equity_curve),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "avg_trade": avg_trade(trades),
        "total_trades": len(trades),
        "final_equity": equity_curve.iloc[-1] if len(equity_curve) > 0 else 0.0,
        "total_return": (
            (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0.0
        ),
    }

    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    """
    Format metrics as a readable string.

    Args:
        metrics: Dict of metrics from calculate_all_metrics()

    Returns:
        Formatted string
    """
    lines = [
        "=" * 50,
        "BACKTEST PERFORMANCE METRICS",
        "=" * 50,
        f"CAGR:              {metrics['cagr']:>10.2%}",
        f"Total Return:      {metrics['total_return']:>10.2%}",
        f"Sharpe Ratio:      {metrics['sharpe']:>10.2f}",
        f"Sortino Ratio:     {metrics['sortino']:>10.2f}",
        f"Calmar Ratio:      {metrics['calmar']:>10.2f}",
        f"Max Drawdown:      {metrics['max_drawdown']:>10.2%}",
        f"Annual Volatility: {metrics['annual_vol']:>10.2%}",
        "-" * 50,
        f"Total Trades:      {metrics['total_trades']:>10}",
        f"Win Rate:          {metrics['win_rate']:>10.2%}",
        f"Profit Factor:     {metrics['profit_factor']:>10.2f}",
        f"Avg Trade:         ${metrics['avg_trade']:>9,.2f}",
        f"Turnover:          {metrics['turnover']:>10.2f}x",
        "-" * 50,
        f"Final Equity:      ${metrics['final_equity']:>9,.2f}",
        "=" * 50,
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics with synthetic data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    equity = pd.Series(100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01), index=dates)
    equity = equity.clip(lower=50000)  # Ensure positive

    returns = equity.pct_change().fillna(0)

    trades = pd.DataFrame(
        {
            "pnl": np.random.randn(100) * 1000,
            "notional": np.abs(np.random.randn(100)) * 10000,
        }
    )

    metrics = calculate_all_metrics(equity, returns, trades)
    print(format_metrics(metrics))
