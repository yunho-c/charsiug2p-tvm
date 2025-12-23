from __future__ import annotations

from dataclasses import asdict

import typer
from rich.console import Console
from rich.table import Table

from charsiug2p_tvm import __version__
from charsiug2p_tvm.config import DEFAULT_CONFIG
from charsiug2p_tvm.harness import reference_g2p

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
def run_model(
    words: list[str] = typer.Argument(..., help="Words to convert to phonemes."),
    lang: str = typer.Option(..., help="Language code (e.g., eng-us)."),
    checkpoint: str = typer.Option(DEFAULT_CONFIG.checkpoint, help="HF checkpoint to use."),
    max_input_bytes: int = typer.Option(DEFAULT_CONFIG.max_input_bytes, help="Max bytes for prefixed word."),
    max_output_len: int = typer.Option(DEFAULT_CONFIG.max_output_len, help="Max output length."),
    space_after_colon: bool = typer.Option(False, help="Insert a space after the language prefix."),
    device: str = typer.Option("cpu", help="Torch device to run on."),
) -> None:
    """Run a reference G2P inference using transformers."""
    phones = reference_g2p(
        words,
        lang,
        checkpoint=checkpoint,
        max_input_bytes=max_input_bytes,
        max_output_len=max_output_len,
        space_after_colon=space_after_colon,
        device=device,
    )
    table = Table(title="CharsiuG2P Reference Output", show_header=True, header_style="bold")
    table.add_column("Word", style="cyan")
    table.add_column("Phonemes", style="white")
    for word, phoneme in zip(words, phones):
        table.add_row(word, phoneme)
    console.print(table)


def cli() -> None:
    app()
