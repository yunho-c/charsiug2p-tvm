# Rust + Flutter (Flutter Rust Bridge) G2P Pipeline Plan

This plan outlines how to scaffold, implement, and test a Rust-based runtime for the CharsiuG2P TVM pipeline with Flutter Rust Bridge (FRB) for mobile use.

## Goals and scope

- Provide a Rust API that mirrors the Python pipeline:
  - [x] Prefix words with "<lang>:"
  - [x] Tokenize with ByT5 tokenizer (google/byt5-small)
  - [x] Run TVM-compiled encoder/decoder (or prefill/step for KV-cache)
  - [x] Greedy decode and return phoneme strings
  - [x] KV-cache path uses decoder_prefill/decoder_step outputs (logits, past_k, past_v, cur_pos) to step tokens one at a time
- Target Flutter (iOS + Android) via FRB.
- Reuse compiled artifacts from `python/src/charsiug2p_tvm/tvm_compile.py`.

## Key decisions

- Runtime strategy (current):
  - Use Relax VM bytecode executables and load them via `vm_load_executable` + `vm_initialization` from Rust (tvm-ffi).
  - AOT remains a future optimization path, not a parallel implementation track.
- Tokenizer strategy:
  - Support ByT5 only.
  - Export tokenizer assets via Python `AutoTokenizer.save_pretrained(...)` and bundle the resulting files.
  - ByT5 does not ship a vocab/model file; it tokenizes raw UTF-8 bytes with a fixed offset (pad/eos/unk => offset=3), so no `tokenizer.json` is expected.
- Artifact metadata:
  - `compile_metadata.txt` is already emitted alongside compiled artifacts (we can migrate to JSON later if desired).
- Artifact discovery (Rust):
  - `g2p_pipeline::ArtifactResolver` auto-derives tokenizer metadata and TVM artifact paths using the `dist/{tokenizers,tvm}/<checkpoint_sanitized>` layout.
  - `ArtifactRoots` supports separate tokenizer/tvm roots so mobile apps can pass per-platform asset storage directories instead of relying on repo-relative paths.
  - CLI honors `CHARSIUG2P_ASSET_ROOT`, `CHARSIUG2P_TOKENIZER_ROOT`, and `CHARSIUG2P_TVM_ROOT` to keep mobile path injection simple.
  - Repo root has a `dist` symlink to `python/dist` so Rust tooling matches the Python default layout.
  - iOS bundles use `target=metal-ios`, so artifacts should live under `dist/tvm/<checkpoint>/b*_in*_out*/metal-ios/`.

## FRB API (initial implementation)

- New crate: `rust/g2p_ffi` exposes Flutter Rust Bridge APIs with an opaque `G2pModel`.
- `G2pModelConfig` fields: `asset_root`, `checkpoint`, `target`, `max_input_bytes`, `max_output_len`, `batch_size`, `tvm_ext`, `use_kv_cache`, `device`, `device_id`, `tokenizer_root`, `tvm_root`.
- Defaults:
  - `checkpoint=charsiu/g2p_multilingual_byT5_tiny_8_layers_100`
  - `target=metal-ios` on iOS, `vulkan` on Android, `llvm` elsewhere
  - `use_kv_cache=true`, `device` inferred from `target` when unset
- Tokenizer is ByT5-only: `g2p_ffi` validates tokenizer metadata (`byt5_offset` or tokenizer name containing `byt5`) and errors otherwise.
- Errors are structured as `G2pFfiError { kind, message, details }` with `G2pErrorKind` (config, artifact, tokenizer, tvm, device, inference).
- `g2p_ffi` builds `cdylib` + `staticlib` outputs for Flutter (in addition to the Rust `lib` artifact).
- FRB API definitions live in `rust/g2p_ffi/src/api.rs`; `rust/g2p_ffi/src/lib.rs` only wires `pub mod api;` and `mod frb_generated;`.
- Codegen config: `flutter/flutter_rust_bridge.yaml`, outputs `flutter/lib/src/{api.dart,frb_generated*.dart}` and `rust/g2p_ffi/src/frb_generated.rs`.
- Codegen command: `flutter_rust_bridge_codegen generate --config-file flutter/flutter_rust_bridge.yaml`.
- Codegen requires Flutter/Dart tooling plus `tvm-ffi-config` on `PATH` (e.g., `python/.pixi/envs/default/bin`) since it runs `cargo expand` on the Rust crate.

## Scaffolding (repo layout)

- `rust/` (Cargo workspace)
  - `g2p_core/`: core pipeline (prefixing, tokenization, padding, decode loop)
  - `g2p_tvm/`: TVM runtime bindings (tvm-ffi) + Relax VM loader helpers
  - `g2p_pipeline/`: TVM + tokenizer pipeline (cacheless + KV-cache)
  - `g2p_ffi/`: FRB API layer exposing Rust functions to Flutter (planned)
  - `g2p_cli/`: lightweight CLI for local testing
- `flutter/` (Flutter app or plugin)
  - `assets/`: tokenizer files + compiled TVM artifacts + manifest
  - FRB-generated bindings + build scripts

## Implementation phases

1) Bootstrap + assets
   - [x] Export tokenizer artifacts from Python and place under `flutter/assets/`.
   - Compile TVM artifacts for:
     - [x] dev: `llvm` (desktop validation)
     - [x] ios: `metal` (arm64)
     - [ ] android: `vulkan` or `opencl` (plus CPU fallback)
   - [x] Write manifest describing shape limits and artifact layout (currently `compile_metadata.txt`).

2) Core Rust pipeline (`g2p_core`)
   - [x] Prefix + UTF-8 byte-length checks (match `max_input_bytes` behavior).
   - [x] Tokenize + pad to static shape (max_input_bytes / max_output_len).
   - [x] Greedy decode loop:
     - [x] Cacheless path (decoder full length) first.
     - [x] KV-cache path using prefill/step second.
   - [x] Decode token IDs to strings, strip special tokens.

3) TVM runtime integration (`g2p_tvm` + tvm-ffi)
   - Provide Rust-safe calls for:
     - [x] Load module
     - [x] Create VM
     - [x] Call "main" (encoder, decoder, prefill, step)
     - [x] Manage device selection and tensors
  - Device type must match the compiled target (e.g., Metal artifacts require Metal tensors and VM initialization on Metal).
   - [x] Start with CPU path (llvm) for validation, then GPU targets.

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

- Decide VM vs AOT runtime path after a quick prototype.
- Choose GPU backend defaults per platform (Metal vs Vulkan/OpenCL).
