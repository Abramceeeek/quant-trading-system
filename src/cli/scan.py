"""
CLI command to scan and validate YAML strategies.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.strategies.loader import list_strategies

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def strategies(directory: str = "strategies") -> None:
    """
    List and validate all YAML strategies in a directory.

    Args:
        directory: Directory to scan for YAML files (default: strategies/)
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        console.print(f"[red][X] Directory not found: {directory}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Scanning strategies in:[/bold] {directory}\n")

    strategies_list = list_strategies(directory)

    if not strategies_list:
        console.print("[yellow]No YAML strategy files found.[/yellow]")
        raise typer.Exit(code=0)

    # Create table
    table = Table(title="YAML Strategies")
    table.add_column("Status", style="green", width=8)
    table.add_column("File", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Type", style="blue")
    table.add_column("Universe")
    table.add_column("Validation")

    all_valid = True

    for strat in strategies_list:
        status = "[OK]" if strat["valid"] else "[X]"
        status_style = "green" if strat["valid"] else "red"

        table.add_row(
            f"[{status_style}]{status}[/{status_style}]",
            strat["file"],
            strat["name"],
            strat["type"],
            strat["universe"],
            strat["validation_msg"],
        )

        if not strat["valid"]:
            all_valid = False

    console.print(table)

    console.print(f"\n[bold]Total strategies:[/bold] {len(strategies_list)}")

    if all_valid:
        console.print("[green][OK] All strategies are valid[/green]")
        raise typer.Exit(code=0)
    else:
        console.print("[red][X] Some strategies have errors[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
