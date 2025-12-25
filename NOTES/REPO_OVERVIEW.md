# charsiug2p-tvm repo overview

This repo is a multi-language port of CharsiuG2P inference to TVM, with Python tooling for export/compile/validation, a Rust runtime pipeline for deployment, and Flutter bindings for mobile apps. The upstream training code/data lives in `external/CharsiuG2P`, and TVM itself is vendored under `external/tvm`.

## Top-level components

- `python/` is the authoring/validation layer. It exports ByT5 tokenizers, compiles Torch → Relax modules (with optional KV-cache prefill/step), and runs reference vs TVM inference/profiling via a Typer CLI.
- `rust/` is the production runtime pipeline. It loads tokenizer metadata + TVM artifacts, performs greedy decode (with or without KV-cache), and exposes a CLI for local testing.
- `flutter/` wraps the Rust pipeline via Flutter Rust Bridge (FRB) and resolves platform-specific asset roots for mobile integration.

## Python tooling flow

- `python/src/charsiug2p_tvm/cli.py` defines CLI commands: export tokenizers, compile TVM artifacts, run TVM inference, verify against the reference, and profile performance.
- `python/src/charsiug2p_tvm/tvm_compile.py` loads `transformers.T5ForConditionalGeneration`, exports encoder/decoder (or encoder + decoder_prefill/decoder_step for KV-cache) with `torch.export`, lowers to Relax, then builds VM executables with `relax.build`. Artifacts are written to `dist/tvm/...` with a `compile_metadata.txt` plus `.relax.py` dumps.
- `python/src/charsiug2p_tvm/tvm_runtime.py` loads VM modules and performs greedy decode, either cacheless (full decoder each step) or KV-cache (prefill + step). It supports multi-batch selection and timed profiling paths.
- `python/src/charsiug2p_tvm/harness.py` is the reference HF path used for correctness checks (`transformers` generate).

## Rust pipeline flow

- `rust/g2p_core` handles `<lang>:` prefixing and input byte-length validation.
- `rust/g2p_tokenizer` loads `tokenizer_metadata.json` and either uses `tokenizers` (via `tokenizer.json`) or a native ByT5 byte tokenizer fallback.
- `rust/g2p_tvm` loads Relax VM modules via `tvm-ffi`, handles device selection, tensor allocation/copies (CPU/GPU), and exposes encoder/decoder (plus KV-cache entrypoints).
- `rust/g2p_pipeline` ties tokenizer + TVM together and runs greedy decoding; `rust/g2p_cli` adds artifact auto-discovery and batch-size selection.

## Flutter bridge

- `flutter/rust/src/api.rs` exposes `g2p_model_new` and `g2p_model_run` over FRB, using the same artifact layout and pipeline logic as the CLI. Defaults pick `metal-ios` on iOS, `vulkan` on Android, and `llvm` elsewhere.

## Artifact layout (shared across Python/Rust/Flutter)

- Tokenizers: `dist/tokenizers/<checkpoint_sanitized>/in{max_input_bytes}_out{max_output_len}/tokenizer_metadata.json` plus tokenizer files.
- TVM artifacts: `dist/tvm/<checkpoint_sanitized>/b{batch_size}_in{max_input_bytes}_out{max_output_len}/{target}/encoder.{ext}`, `decoder.{ext}`, and optional `decoder_prefill.{ext}`/`decoder_step.{ext}` for KV-cache.
- Rust/Flutter resolve these via `ArtifactResolver` or explicit roots (`CHARSIUG2P_ASSET_ROOT`, `CHARSIUG2P_TOKENIZER_ROOT`, `CHARSIUG2P_TVM_ROOT` on the CLI side; `G2pModelConfig` in Flutter).

## Key defaults and assumptions

- Default checkpoint: `charsiu/g2p_multilingual_byT5_tiny_8_layers_100`.
- Default bounds: `max_input_bytes=64`, `max_output_len=128`, batch size 1.
- Prefix behavior: `<lang>:` is inserted without a trailing space by default (toggle via `space_after_colon` in both Python and Rust/Flutter).
