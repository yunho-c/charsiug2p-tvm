# CharsiuG2P Rust Workspace

This directory contains the Rust implementation of the CharsiuG2P TVM pipeline plus a CLI for local testing. It loads tokenizer metadata exported from the Python tooling and runs TVM-compiled encoder/decoder modules with either cacheless greedy decode or KV-cache artifacts (`decoder_prefill`/`decoder_step`).

## Crates

- `g2p_core`: prefixing logic and input length validation.
- `g2p_tokenizer`: ByT5 byte tokenizer and `tokenizer.json` backend support.
- `g2p_tvm`: TVM module loader and tensor helpers via `tvm-ffi`.
- `g2p_pipeline`: pipeline wiring plus artifact auto-discovery helpers.
- `g2p_cli`: CLI entrypoint for local inference.

The Flutter Rust Bridge layer lives at `flutter/rust`.

## Artifact layout

The Rust pipeline expects the same layout that the Python tooling produces under `dist/`. The repo root includes a `dist` symlink to `python/dist` for convenience. If the symlink is missing, the CLI still checks `python/dist` and parent directories when auto-deriving paths.

Tokenizer assets:

`dist/tokenizers/<checkpoint_sanitized>/in{max_input_bytes}_out{max_output_len}/tokenizer_metadata.json`

TVM artifacts:

`dist/tvm/<checkpoint_sanitized>/b{batch_size}_in{max_input_bytes}_out{max_output_len}/{target}/encoder.{ext}`
`dist/tvm/<checkpoint_sanitized>/b{batch_size}_in{max_input_bytes}_out{max_output_len}/{target}/decoder.{ext}`
`dist/tvm/<checkpoint_sanitized>/b{batch_size}_in{max_input_bytes}_out{max_output_len}/{target}/decoder_prefill.{ext}`
`dist/tvm/<checkpoint_sanitized>/b{batch_size}_in{max_input_bytes}_out{max_output_len}/{target}/decoder_step.{ext}`

`checkpoint_sanitized` replaces `/` with `_`.
For iOS, we use `target=metal-ios` so bundle the artifacts under a `metal-ios/` directory to keep target/device mismatches obvious.

## Running the CLI

The CLI auto-derives tokenizer and TVM paths unless you pass them explicitly. It also respects `CHARSIUG2P_ASSET_ROOT`, `CHARSIUG2P_TOKENIZER_ROOT`, and `CHARSIUG2P_TVM_ROOT` to make mobile asset injection easy.

Example (from repo root):

```bash
PATH="python/.pixi/envs/default/bin:$PATH" \
LD_LIBRARY_PATH="python/.pixi/envs/default/lib/python3.12/site-packages/tvm_ffi/lib:$LD_LIBRARY_PATH" \
cargo run -p charsiug2p-g2p-cli -- \
  --checkpoint charsiu/g2p_multilingual_byT5_tiny_8_layers_100 \
  --tvm-target llvm \
  --max-input-bytes 64 \
  --max-output-len 128 \
  --lang eng-us \
  Char siu
```

You may need to use absolute paths:

```bash
REPO=/home/ycho358/GitHub/charsiug2p-tvm
PATH="$REPO/python/.pixi/envs/default/bin:$PATH" \
LD_LIBRARY_PATH="$REPO/python/.pixi/envs/default/lib/python3.12/site-packages/tvm_ffi/lib:$REPO/external/tvm/build:$LD_LIBRARY_PATH" \
LD_PRELOAD="$REPO/external/tvm/build/libtvm_runtime.so" \
cargo run ...
```

To bypass auto-discovery:

```bash
cargo run -p charsiug2p-g2p-cli -- \
  --tokenizer-metadata dist/tokenizers/.../tokenizer_metadata.json \
  --encoder dist/tvm/.../encoder.so \
  --decoder dist/tvm/.../decoder.so \
  --lang eng-us \
  Char siu
```

KV-cache is the default; ensure `decoder_prefill.{ext}` and `decoder_step.{ext}` are present in the same target directory (or pass `--decoder-prefill`/`--decoder-step` explicitly). Use `--kv-cache=false` to run the cacheless decoder.
The CLI defaults `--device` based on `--tvm-target` (for example, `metal` → `metal`, `llvm` → `cpu`). Override with `--device` and `--device-id` when needed.

To auto-select from multiple compiled batch sizes, use `--batch-sizes` (comma-separated). The CLI will pick the smallest batch size that fits the remaining words.

## Development notes

- Build requires `tvm-ffi-config` on `PATH` (installable via `pip install` in the pixi environment).
- Runtime needs `tvm_ffi` shared libraries on the platform library path (`LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`).
- Use pixi's Python environment for any scripts or builds that touch TVM.

## Tests

```bash
PATH="python/.pixi/envs/default/bin:$PATH" cargo test
```
