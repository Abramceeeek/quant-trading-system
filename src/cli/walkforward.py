"""CLI command for walk-forward analysis (placeholder)."""

import typer
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def run(strategy_file: str, folds: int = 5) -> None:
    """Run walk-forward analysis (to be implemented)."""
    console.print("[yellow]Walk-forward analysis not yet implemented[/yellow]")
    console.print(f"Will split data into {folds} sequential folds")


if __name__ == "__main__":
    app()
