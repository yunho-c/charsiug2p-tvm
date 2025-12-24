from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from charsiug2p_tvm import __version__
from charsiug2p_tvm.config import DEFAULT_CONFIG, TARGET_CONFIGS, default_device_for_target, resolve_target
from charsiug2p_tvm.harness import reference_g2p
from charsiug2p_tvm.tvm_compile import compile_tvm_module, default_output_dir
from charsiug2p_tvm.tvm_runtime import tvm_g2p, tvm_g2p_cached
from charsiug2p_tvm.eval import evaluate_against_reference, prepare_samples
from charsiug2p_tvm.profile import parse_targets, profile_targets, write_profile_csv

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
def compile_model(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory for compiled artifacts (defaults to dist/tvm/<model>/<details>/<target>).",
    ),
    checkpoint: str = typer.Option(DEFAULT_CONFIG.checkpoint, help="HF checkpoint to use."),
    batch_size: int = typer.Option(DEFAULT_CONFIG.batch_size, help="Batch size to compile for."),
    max_input_bytes: int = typer.Option(DEFAULT_CONFIG.max_input_bytes, help="Max bytes for prefixed word."),
    max_output_len: int = typer.Option(DEFAULT_CONFIG.max_output_len, help="Max output length."),
    target: str = typer.Option(
        "llvm",
        help=f"TVM target or preset ({', '.join(TARGET_CONFIGS)}).",
    ),
    output_ext: str | None = typer.Option(None, help="Output extension (defaults by target)."),
    mixed_precision: bool = typer.Option(
        False,
        "--mixed-precision/--no-mixed-precision",
        help="Enable Relax mixed-precision rewriting.",
    ),
    mixed_precision_out_dtype: str = typer.Option(
        "float32",
        "--mixed-precision-out-dtype",
        help="Accumulator dtype for mixed precision (e.g., float32, float16).",
    ),
    fp16_input_names: list[str] | None = typer.Option(
        None,
        "--fp16-input",
        help="Function parameter name to cast to fp16 (repeatable).",
    ),
    use_kv_cache: bool = typer.Option(
        False,
        "--kv-cache/--no-kv-cache",
        help="Compile experimental KV-cache prefill/step modules.",
    ),
) -> None:
    """Compile encoder/decoder modules into TVM runtime artifacts."""
    resolved = resolve_target(target, output_ext=output_ext)
    output_ext = resolved.output_ext
    if output_dir is None:
        output_dir = default_output_dir(
            checkpoint=checkpoint,
            target=resolved.name,
            batch_size=batch_size,
            max_input_bytes=max_input_bytes,
            max_output_len=max_output_len,
        )
    console.print(f"[cyan]Compiling TVM artifacts for {checkpoint}...[/cyan]")
    artifacts = compile_tvm_module(
        output_dir=output_dir,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_input_bytes=max_input_bytes,
        max_output_len=max_output_len,
        target=target,
        output_ext=output_ext,
        mixed_precision=mixed_precision,
        mixed_precision_out_dtype=mixed_precision_out_dtype,
        fp16_input_names=fp16_input_names,
        use_kv_cache=use_kv_cache,
    )
    table = Table(title="TVM Compile Outputs", show_header=True, header_style="bold")
    table.add_column("Module", style="cyan")
    table.add_column("Path", style="white")
    for name, path in artifacts.items():
        table.add_row(name, str(path))
    console.print(table)


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


@app.command("run-tvm")
def run_tvm_model(
    words: list[str] = typer.Argument(..., help="Words to convert to phonemes."),
    lang: str = typer.Option(..., help="Language code (e.g., eng-us)."),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory containing compiled artifacts (defaults to dist/tvm/<model>/<details>/<target>).",
    ),
    checkpoint: str = typer.Option(DEFAULT_CONFIG.checkpoint, help="HF checkpoint to use."),
    target: str = typer.Option(
        "llvm",
        help=f"TVM target or preset used to compile artifacts ({', '.join(TARGET_CONFIGS)}).",
    ),
    output_ext: str | None = typer.Option(None, help="Artifact extension (defaults by target)."),
    batch_size: int = typer.Option(DEFAULT_CONFIG.batch_size, help="Batch size to run."),
    max_input_bytes: int = typer.Option(DEFAULT_CONFIG.max_input_bytes, help="Max bytes for prefixed word."),
    max_output_len: int = typer.Option(DEFAULT_CONFIG.max_output_len, help="Max output length."),
    space_after_colon: bool = typer.Option(False, help="Insert a space after the language prefix."),
    device: str | None = typer.Option(
        None,
        help="TVM device string (e.g., cpu, cuda, metal). Defaults by target.",
    ),
    use_kv_cache: bool = typer.Option(
        False,
        "--kv-cache/--no-kv-cache",
        help="Use experimental KV-cache prefill/step artifacts.",
    ),
) -> None:
    """Run G2P inference using compiled TVM artifacts."""
    if device is None:
        device = default_device_for_target(target)
    runner = tvm_g2p_cached if use_kv_cache else tvm_g2p
    phones = runner(
        words,
        lang,
        output_dir=output_dir,
        checkpoint=checkpoint,
        target=target,
        output_ext=output_ext,
        batch_size=batch_size,
        max_input_bytes=max_input_bytes,
        max_output_len=max_output_len,
        space_after_colon=space_after_colon,
        device=device,
    )
    table = Table(title="CharsiuG2P TVM Output", show_header=True, header_style="bold")
    table.add_column("Word", style="cyan")
    table.add_column("Phonemes", style="white")
    for word, phoneme in zip(words, phones):
        table.add_row(word, phoneme)
    console.print(table)


@app.command("verify")
def verify_tvm(
    data_path: Path = typer.Argument(..., help="TSV file or directory to evaluate."),
    lang: str | None = typer.Option(None, help="Override language code for a single TSV file."),
    limit: int | None = typer.Option(None, help="Limit the number of samples."),
    shuffle: bool = typer.Option(False, help="Shuffle samples before limiting."),
    seed: int | None = typer.Option(None, help="Shuffle seed."),
    strip_spaces: bool = typer.Option(False, help="Remove spaces before comparison."),
    checkpoint: str = typer.Option(DEFAULT_CONFIG.checkpoint, help="HF checkpoint to use."),
    max_input_bytes: int = typer.Option(DEFAULT_CONFIG.max_input_bytes, help="Max bytes for prefixed word."),
    max_output_len: int = typer.Option(DEFAULT_CONFIG.max_output_len, help="Max output length."),
    space_after_colon: bool = typer.Option(False, help="Insert a space after the language prefix."),
    device: str = typer.Option("cpu", help="Device for reference run."),
    tvm_output_dir: Path | None = typer.Option(
        None,
        "--tvm-output-dir",
        help="Directory containing compiled artifacts (defaults to dist/tvm/<model>/<details>/<target>).",
    ),
    tvm_target: str = typer.Option(
        "llvm",
        help=f"TVM target or preset used to compile artifacts ({', '.join(TARGET_CONFIGS)}).",
    ),
    tvm_output_ext: str | None = typer.Option(None, help="Artifact extension (defaults by target)."),
    tvm_batch_size: int = typer.Option(DEFAULT_CONFIG.batch_size, help="Compiled TVM batch size."),
    tvm_device: str | None = typer.Option(
        None,
        help="TVM device string (e.g., cpu, cuda, metal). Defaults by target.",
    ),
    ref_batch_size: int = typer.Option(8, help="Reference batch size."),
    use_kv_cache: bool = typer.Option(
        False,
        "--kv-cache/--no-kv-cache",
        help="Use KV-cache prefill/step artifacts for TVM evaluation.",
    ),
) -> None:
    """Compare TVM outputs against the reference transformers path."""
    if tvm_device is None:
        tvm_device = default_device_for_target(tvm_target)
    samples = prepare_samples(
        path=data_path,
        language=lang,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
    )
    metrics = evaluate_against_reference(
        samples,
        checkpoint=checkpoint,
        max_input_bytes=max_input_bytes,
        max_output_len=max_output_len,
        space_after_colon=space_after_colon,
        strip_spaces=strip_spaces,
        tvm_output_dir=tvm_output_dir,
        tvm_target=tvm_target,
        tvm_output_ext=tvm_output_ext,
        tvm_batch_size=tvm_batch_size,
        ref_batch_size=ref_batch_size,
        ref_device=device,
        tvm_device=tvm_device,
        use_kv_cache=use_kv_cache,
    )
    table = Table(title="TVM vs Reference Metrics", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Samples", str(metrics.total))
    table.add_row("Exact match", f"{metrics.exact_match} ({metrics.exact_match_rate:.2%})")
    table.add_row("CER", f"{metrics.cer:.4f}")
    console.print(table)


@app.command("profile")
def profile_tvm(
    data_path: Path = typer.Argument(..., help="TSV file or directory to profile."),
    targets: list[str] = typer.Option(
        ...,
        "--target",
        help=f"TVM target(s) or presets to profile ({', '.join(TARGET_CONFIGS)}).",
    ),
    limit: int | None = typer.Option(None, help="Limit the number of samples."),
    shuffle: bool = typer.Option(False, help="Shuffle samples before limiting."),
    seed: int | None = typer.Option(None, help="Shuffle seed."),
    checkpoint: str = typer.Option(DEFAULT_CONFIG.checkpoint, help="HF checkpoint to use."),
    max_input_bytes: int = typer.Option(DEFAULT_CONFIG.max_input_bytes, help="Max bytes for prefixed word."),
    max_output_len: int = typer.Option(DEFAULT_CONFIG.max_output_len, help="Max output length."),
    space_after_colon: bool = typer.Option(False, help="Insert a space after the language prefix."),
    tvm_output_ext: str | None = typer.Option(None, help="Artifact extension (defaults by target)."),
    tvm_batch_size: int = typer.Option(DEFAULT_CONFIG.batch_size, help="Compiled TVM batch size."),
    runs: int = typer.Option(1, help="Number of timed runs to average."),
    warmup: bool = typer.Option(True, help="Run one warmup pass before timing."),
    use_kv_cache: bool = typer.Option(
        False,
        "--kv-cache/--no-kv-cache",
        help="Use KV-cache prefill/step artifacts for profiling.",
    ),
    device: str | None = typer.Option(None, help="Override device for all targets."),
    output_file: Path = typer.Option(
        Path("dist/profile_results.csv"),
        help="CSV output path for profiling results.",
    ),
) -> None:
    """Profile TVM inference across multiple targets and write a CSV report."""
    normalized_targets = parse_targets(targets)
    results = profile_targets(
        data_path=data_path,
        targets=normalized_targets,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
        checkpoint=checkpoint,
        max_input_bytes=max_input_bytes,
        max_output_len=max_output_len,
        space_after_colon=space_after_colon,
        tvm_output_ext=tvm_output_ext,
        tvm_batch_size=tvm_batch_size,
        runs=runs,
        warmup=warmup,
        device=device,
        use_kv_cache=use_kv_cache,
    )
    if not results:
        console.print("[yellow]No samples found; nothing to profile.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="TVM Profile Results", show_header=True, header_style="bold")
    table.add_column("Target", style="cyan")
    table.add_column("Device", style="white")
    table.add_column("Samples", style="white", justify="right")
    table.add_column("Total (s)", style="white", justify="right")
    table.add_column("Encoder (s)", style="white", justify="right")
    table.add_column("Decoder (s)", style="white", justify="right")
    table.add_column("Per-sample (ms)", style="white", justify="right")
    table.add_column("Encoder/sample (ms)", style="white", justify="right")
    table.add_column("Decoder/sample (ms)", style="white", justify="right")
    table.add_column("Steps/sample", style="white", justify="right")
    table.add_column("Decode ms/step", style="white", justify="right")
    for result in results:
        table.add_row(
            result.target,
            result.device,
            str(result.samples),
            f"{result.total_seconds:.3f}",
            f"{result.encoder_seconds:.3f}",
            f"{result.decoder_seconds:.3f}",
            f"{result.per_sample_ms:.3f}",
            f"{result.encoder_per_sample_ms:.3f}",
            f"{result.decoder_per_sample_ms:.3f}",
            f"{result.decode_steps_per_sample:.3f}",
            f"{result.decode_ms_per_step:.4f}",
        )
    console.print(table)

    write_profile_csv(output_file, results)
    console.print(f"[green]Saved CSV to {output_file}[/green]")


def cli() -> None:
    app()
