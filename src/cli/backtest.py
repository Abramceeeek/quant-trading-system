"""
CLI command to run backtests.
"""

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from src.backtest.engine import BacktestEngine
from src.core.config import config
from src.core.universes import get_universe
from src.datasource.yfinance_source import YFinanceSource
from src.strategies.loader import load_strategy_from_yaml

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def run(
    strategy_file: str,
    start: str = "2020-01-01",
    end: str = "",
    report: bool = False,
) -> None:
    """
    Run backtest for a YAML strategy.

    Args:
        strategy_file: Path to YAML strategy file
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD, default: today)
        report: Generate detailed report with plots
    """
    console.print(f"[bold]Loading strategy:[/bold] {strategy_file}")

    try:
        strategy, spec = load_strategy_from_yaml(strategy_file)
    except Exception as e:
        console.print(f"[red][X] Failed to load strategy:[/red] {e}")
        raise typer.Exit(code=1)

    # Get universe
    universe_name = spec.get("universe", "demo")
    universe = get_universe(universe_name)
    console.print(f"[bold]Universe:[/bold] {universe_name} ({len(universe)} symbols)")

    # Setup datasource and engine
    datasource = YFinanceSource()

    backtest_config = spec.get("backtest", {})
    sizing_config = spec.get("sizing", {})

    engine = BacktestEngine(
        datasource=datasource,
        initial_capital=backtest_config.get("initial_capital", 100_000),
        commission_pct=backtest_config.get("commission_pct", 0.001),
        slippage_pct=backtest_config.get("slippage_pct", 0.0005),
        sizing_method=sizing_config.get("method", "equal_weight"),
        max_positions=sizing_config.get("max_positions", 20),
    )

    # Run backtest
    if not end:
        end = datetime.now().strftime("%Y-%m-%d")

    console.print(f"[bold]Running backtest:[/bold] {start} to {end}")

    try:
        results = engine.run(strategy, universe, start, end)
    except Exception as e:
        console.print(f"[red][X] Backtest failed:[/red] {e}")
        raise typer.Exit(code=1)

    # Print results
    engine.print_results()

    # Save results if requested
    if report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = spec.get("strategy", {}).get("name", "strategy")
        output_dir = config.reports_dir / timestamp / strategy_name

        engine.save_results(output_dir)
        console.print(f"\n[green][OK][/green] Report saved to: {output_dir}")

    console.print("\n[green][OK] Backtest complete[/green]")


if __name__ == "__main__":
    app()
