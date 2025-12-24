# Rust + Flutter (Flutter Rust Bridge) G2P Pipeline Plan

This plan outlines how to scaffold, implement, and test a Rust-based runtime for the CharsiuG2P TVM pipeline with Flutter Rust Bridge (FRB) for mobile use.

## Goals and scope

- Provide a Rust API that mirrors the Python pipeline:
  - Prefix words with "<lang>:"
  - Tokenize with ByT5 tokenizer (google/byt5-small)
  - Run TVM-compiled encoder/decoder (or prefill/step for KV-cache)
  - Greedy decode and return phoneme strings
- Target Flutter (iOS + Android) via FRB.
- Reuse compiled artifacts from `python/src/charsiug2p_tvm/tvm_compile.py`.

## Key decisions (to lock in early)

- Runtime strategy:
  - Option A (recommended first): Relax VM with a thin C++ wrapper that exposes a C ABI to Rust for load + invoke.
  - Option B: Compile to AOT and call directly via TVM C runtime (less VM logic).
- Tokenizer strategy:
  - Prefer HuggingFace `tokenizers` crate if `tokenizer.json` is available.
  - If only `tokenizer.model` exists, use a SentencePiece-capable Rust crate.
  - Export tokenizer assets via Python `AutoTokenizer.save_pretrained(...)` and bundle the resulting files.
- Artifact metadata:
  - Add a small manifest (JSON) alongside compiled artifacts capturing: checkpoint, max_input_bytes, max_output_len, batch_size, dtype, target, output_ext, use_kv_cache, tokenizer files, and a version.

## Scaffolding (repo layout)

- `rust/` (Cargo workspace)
  - `g2p_core/`: core pipeline (prefixing, tokenization, padding, decode loop)
  - `tvm_sys/`: low-level FFI bindings to TVM runtime / VM wrapper
  - `tvm_bridge/` (optional): C++ wrapper with C ABI for Relax VM
  - `g2p_ffi/`: FRB API layer exposing Rust functions to Flutter
  - `g2p_cli/`: lightweight CLI for local testing
- `mobile/` (Flutter app or plugin)
  - `assets/`: tokenizer files + compiled TVM artifacts + manifest
  - FRB-generated bindings + build scripts

## Implementation phases

1) Bootstrap + assets
   - Export tokenizer artifacts from Python and place under `mobile/assets/`.
   - Compile TVM artifacts for:
     - dev: `llvm` (desktop validation)
     - ios: `metal` (arm64)
     - android: `vulkan` or `opencl` (plus CPU fallback)
   - Write manifest describing shape limits and artifact layout.

2) Core Rust pipeline (`g2p_core`)
   - Prefix + UTF-8 byte-length checks (match `max_input_bytes` behavior).
   - Tokenize + pad to static shape (max_input_bytes / max_output_len).
   - Greedy decode loop:
     - Cacheless path (decoder full length) first.
     - KV-cache path using prefill/step second.
   - Decode token IDs to strings, strip special tokens.

3) TVM runtime integration (`tvm_sys` + wrapper)
   - Provide Rust-safe calls for:
     - Load module
     - Create VM
     - Call "main" (encoder, decoder, prefill, step)
     - Manage device selection and tensors
   - Start with CPU path (llvm) for validation, then GPU targets.

4) FRB API (`g2p_ffi`)
   - `G2pModel::new(config)` to load assets + artifacts.
   - `G2pModel::run(words, lang, options)` returns phoneme strings.
   - Async-friendly API for Flutter (avoid blocking UI thread).

5) Packaging + build tooling
   - iOS: build TVM runtime + Rust static lib -> XCFramework.
   - Android: build TVM runtime + Rust cdylib via `cargo-ndk`.
   - Add scripts to copy assets to proper Flutter paths.

## Testing strategy

- Unit tests (Rust):
  - Prefixing rules, byte-length checks
  - Tokenization parity on small fixtures
  - Greedy decode step logic (EOS stop, padding)
- Integration tests:
  - Load CPU artifacts and compare outputs to a golden set.
  - Golden set created via Python harness (store JSON fixture).
- Mobile smoke tests:
  - Simple Flutter UI invoking FRB with a few words.
  - Compare outputs to golden set for 1-2 languages.
- Performance checks:
  - Time encoder/decoder steps and compare to Python baseline.

## Deliverables (first milestones)

- M1: Rust pipeline runs on desktop with llvm artifacts + golden parity.
- M2: iOS + Android builds load artifacts and return correct outputs.
- M3: KV-cache path enabled and measured for speedup.

## Open questions

- Confirm tokenizer format availability for ByT5 (JSON vs SentencePiece).
- Decide VM vs AOT runtime path after a quick prototype.
- Choose GPU backend defaults per platform (Metal vs Vulkan/OpenCL).
