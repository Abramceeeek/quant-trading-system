"""CLI command for live trading (placeholder)."""

import typer
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def run(strategy_file: str, confirm: bool = False) -> None:
    """Run live trading (requires --confirm)."""
    if not confirm:
        console.print("[red]ERROR: Live trading requires --confirm flag[/red]")
        raise typer.Exit(code=1)
    console.print("[yellow]Live trading not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
