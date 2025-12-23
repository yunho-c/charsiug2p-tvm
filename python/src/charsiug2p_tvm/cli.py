from __future__ import annotations

from dataclasses import asdict

import typer
from rich.console import Console
from rich.table import Table

from charsiug2p_tvm import __version__
from charsiug2p_tvm.config import DEFAULT_CONFIG

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _print_config_table() -> None:
    config = asdict(DEFAULT_CONFIG)
    table = Table(title="CharsiuG2P TVM Config", show_header=True, header_style="bold")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    for key, value in config.items():
        table.add_row(str(key), str(value))
    console.print(table)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        console.print(f"charsiug2p-tvm {__version__}")
        raise typer.Exit()


@app.command("info")
def info() -> None:
    """Show the current default configuration."""
    _print_config_table()


@app.command("compile")
def compile_model() -> None:
    """Compile the model with TVM (placeholder)."""
    console.print("[yellow]Compile pipeline not implemented yet.[/yellow]")
    console.print("Start in `charsiug2p_tvm/tvm_compile.py`.")


@app.command("run")
def run_model() -> None:
    """Run a single G2P inference (placeholder)."""
    console.print("[yellow]Run pipeline not implemented yet.[/yellow]")
    console.print("Start in `charsiug2p_tvm/harness.py`.")


def cli() -> None:
    app()
