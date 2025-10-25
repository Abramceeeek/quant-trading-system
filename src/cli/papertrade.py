"""
CLI command for paper trading.
Generates and routes orders to IBKR paper account based on strategy signals.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.core.config import config
from src.core.universes import get_universe
from src.datasource.yfinance_source import YFinanceSource
from src.strategies.loader import load_strategy_from_yaml

console = Console()
app = typer.Typer(add_completion=False)


def _get_latest_signals(
    strategy: Any, prices: dict[str, pd.DataFrame]
) -> tuple[list[str], list[str]]:
    """
    Get latest entry and exit signals from strategy.

    Args:
        strategy: Strategy instance
        prices: Dict of symbol -> price DataFrame

    Returns:
        Tuple of (entry_symbols, exit_symbols)
    """
    entry_symbols = []
    exit_symbols = []

    for symbol, df in prices.items():
        if df.empty or len(df) < 2:
            continue

        try:
            signals = strategy.generate_signals(df)
            if signals.empty:
                continue

            # Check latest signal (today)
            latest_entry = bool(signals["entry"].iloc[-1])
            latest_exit = bool(signals["exit"].iloc[-1])

            if latest_entry:
                entry_symbols.append(symbol)
            if latest_exit:
                exit_symbols.append(symbol)

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to generate signals for {symbol}: {e}[/yellow]")
            continue

    return entry_symbols, exit_symbols


def _calculate_quantities(
    symbols: list[str],
    prices: dict[str, pd.DataFrame],
    notional_per_symbol: float = 0.0,
    fixed_qty: int = 100,
) -> dict[str, int]:
    """
    Calculate order quantities for symbols.

    Args:
        symbols: List of symbols to trade
        prices: Dict of symbol -> price DataFrame
        notional_per_symbol: Target notional value per symbol (overrides fixed_qty if > 0)
        fixed_qty: Fixed number of shares if notional not specified

    Returns:
        Dict of symbol -> quantity
    """
    orders = {}

    for symbol in symbols:
        if symbol not in prices or prices[symbol].empty:
            continue

        try:
            latest_price = float(prices[symbol]["close"].iloc[-1])

            if notional_per_symbol > 0:
                qty = max(1, int(math.floor(notional_per_symbol / latest_price)))
            else:
                qty = fixed_qty

            orders[symbol] = qty

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to calculate qty for {symbol}: {e}[/yellow]")
            continue

    return orders


@app.command()
def run(
    strategy_file: str = typer.Argument(..., help="Path to YAML strategy file"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Dry run mode"),
    qty: int = typer.Option(
        100, "--qty", min=1, help="Fixed shares per symbol (ignored if --notional > 0)"
    ),
    notional: float = typer.Option(
        0.0, "--notional", help="Target notional per symbol in USD (overrides --qty if > 0)"
    ),
    max_symbols: int = typer.Option(10, "--max-symbols", help="Max symbols to trade"),
) -> None:
    """
    Generate paper trading orders from strategy signals.

    Workflow:
    1. Load strategy and fetch latest prices
    2. Generate signals for each symbol
    3. Calculate quantities based on sizing method
    4. Display orders (dry-run) or send to IBKR (live)
    """
    console.print(f"[bold]Paper Trading:[/bold] {strategy_file}")

    # Load strategy
    try:
        strategy, spec = load_strategy_from_yaml(strategy_file)
    except Exception as e:
        console.print(f"[red][X] Failed to load strategy:[/red] {e}")
        raise typer.Exit(code=1)

    # Get universe
    universe_name = spec.get("universe", "demo")
    universe = get_universe(universe_name)
    console.print(f"[bold]Universe:[/bold] {universe_name} ({len(universe)} symbols)")

    # Fetch latest prices
    console.print("[bold]Fetching latest prices...[/bold]")
    datasource = YFinanceSource()

    try:
        # Get last 100 days of data (enough for most indicators)
        price_data = datasource.get_prices(universe, start="2024-09-01", end="", interval="1d")

        if not price_data:
            console.print("[red][X] No price data available[/red]")
            raise typer.Exit(code=1)

        console.print(f"[green]Loaded data for {len(price_data)} symbols[/green]")

    except Exception as e:
        console.print(f"[red][X] Failed to fetch prices:[/red] {e}")
        raise typer.Exit(code=1)

    # Generate signals
    console.print("[bold]Generating signals...[/bold]")
    entry_symbols, exit_symbols = _get_latest_signals(strategy, price_data)

    # Remove overlaps (exit takes precedence)
    exit_set = set(exit_symbols)
    entry_symbols = [s for s in entry_symbols if s not in exit_set]

    # Apply max symbols limit
    entry_symbols = entry_symbols[:max_symbols]
    exit_symbols = exit_symbols[:max_symbols]

    console.print(f"[cyan]Entry signals:[/cyan] {len(entry_symbols)}")
    console.print(f"[magenta]Exit signals:[/magenta] {len(exit_symbols)}")

    if not entry_symbols and not exit_symbols:
        console.print("[yellow]No signals today. Strategy conditions not met.[/yellow]")
        raise typer.Exit(code=0)

    # Calculate quantities
    buy_orders = _calculate_quantities(entry_symbols, price_data, notional, qty)
    sell_orders = _calculate_quantities(exit_symbols, price_data, notional, qty)

    # Convert sells to negative quantities
    sell_orders = {k: -v for k, v in sell_orders.items()}

    # Merge orders
    all_orders = {**buy_orders, **sell_orders}

    # Display orders
    table = Table(title="Paper Trading Orders")
    table.add_column("Symbol", style="cyan")
    table.add_column("Action", style="green")
    table.add_column("Quantity", style="yellow")
    table.add_column("Type", style="blue")
    table.add_column("Status")

    for symbol, qty_signed in all_orders.items():
        action = "BUY" if qty_signed > 0 else "SELL"
        qty_abs = abs(qty_signed)
        status = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]READY[/green]"

        table.add_row(symbol, action, str(qty_abs), "MARKET", status)

    console.print(table)

    # Summary
    total_buy = sum(1 for q in all_orders.values() if q > 0)
    total_sell = sum(1 for q in all_orders.values() if q < 0)
    console.print(f"\n[bold]Total:[/bold] {total_buy} buys, {total_sell} sells")

    if dry_run:
        console.print("\n[green][OK] DRY RUN - No orders sent[/green]")
    else:
        # Confirmation prompt
        confirm = typer.confirm("\nSend orders to IBKR paper account?")
        if not confirm:
            console.print("[yellow]Aborted[/yellow]")
            raise typer.Exit(code=0)

        # Send orders to IBKR
        console.print("\n[bold]Sending orders to IBKR...[/bold]")

        try:
            from src.execution.order_router import route_orders

            results = route_orders(all_orders, dry_run=False)

            # Display results
            results_table = Table(title="Order Execution Results")
            results_table.add_column("Symbol", style="cyan")
            results_table.add_column("Qty", style="yellow")
            results_table.add_column("Status", style="green")
            results_table.add_column("Order ID", style="blue")

            for result in results:
                symbol = result["symbol"]
                qty = result["qty"]
                status = result["status"]
                order_id = result.get("orderId", "N/A")

                # Color code status
                if "ERROR" in status or "FAILED" in status:
                    status_display = f"[red]{status}[/red]"
                elif status in ["Submitted", "PreSubmitted", "Filled"]:
                    status_display = f"[green]{status}[/green]"
                else:
                    status_display = status

                results_table.add_row(
                    symbol,
                    str(qty),
                    status_display,
                    str(order_id) if order_id else "N/A",
                )

            console.print("\n")
            console.print(results_table)

            # Summary
            success_count = sum(
                1 for r in results if "ERROR" not in r["status"] and r["status"] != "SKIPPED"
            )
            error_count = sum(1 for r in results if "ERROR" in r["status"])

            console.print(f"\n[bold]Results:[/bold] {success_count} successful, {error_count} errors")

            if error_count == 0:
                console.print("\n[green][OK] All orders sent successfully[/green]")
            else:
                console.print(
                    "\n[yellow]Warning: Some orders failed. Check TWS/Gateway for details.[/yellow]"
                )

        except Exception as e:
            console.print(f"\n[red][X] Order routing failed:[/red] {e}")
            console.print(
                "[yellow]Make sure TWS/Gateway is running and API is enabled[/yellow]"
            )
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
