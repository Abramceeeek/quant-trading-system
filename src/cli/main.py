"""
Main CLI entry point using Typer.
Provides commands: scan, backtest, walkforward, papertrade, live.
"""

import typer
from rich.console import Console

# Import subcommands
from src.cli import backtest, live, papertrade, scan, walkforward

console = Console()

app = typer.Typer(
    name="qts",
    help="Quant Trading System - YAML-driven backtesting and execution",
    add_completion=False,
)

# Register subcommands
app.add_typer(scan.app, name="scan", help="Scan and validate YAML strategies")
app.add_typer(backtest.app, name="backtest", help="Run backtests")
app.add_typer(walkforward.app, name="walkforward", help="Walk-forward analysis")
app.add_typer(papertrade.app, name="papertrade", help="Paper trading on IBKR")
app.add_typer(live.app, name="live", help="Live trading (guarded)")


@app.command()
def version() -> None:
    """Show version information."""
    console.print("[bold green]Quant Trading System v0.1.0[/bold green]")
    console.print("A clean, staged quantitative trading system")


if __name__ == "__main__":
    app()
