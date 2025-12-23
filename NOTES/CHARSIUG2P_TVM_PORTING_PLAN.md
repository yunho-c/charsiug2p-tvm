# CharsiuG2P -> TVM Porting Plan

## Goal (define the port)

Scope the port as **inference on TVM**: Python tokenizer + `<lang>:` prefixing + TVM-compiled ByT5/T5 encoder-decoder forward passes + greedy decode (beam=1).

Defer training and the custom Switch/MoE training model (`external/CharsiuG2P/src/ByT5_MoE.py`) until after a working baseline, since the released HF checkpoints are standard `T5ForConditionalGeneration`.

## Strategic architecture

```mermaid
flowchart LR
  U[words + lang code] --> P[prefix: "<lang>:" + word]
  P --> T[ByT5 tokenizer (Python)]
  T --> E[TVM Module: encoder]
  T --> D[TVM Module: decoder (step or full)]
  E --> D
  D --> G[greedy loop (Python first)]
  G --> TD[tokenizer.decode (Python)]
  TD --> O[pron strings]
```

## Key TVM approach choices (recommendations)

- Frontend: start with `torch.export.export()` -> `tvm.relax.frontend.torch.from_exported_program(...)`. Keep ONNX as fallback if torch export hits unsupported ops.
- Runtime: start with `relax.VirtualMachine` (fast iteration, fewer AOT constraints). Consider AOT later for embedded use.
- Decode: ship an MVP without KV-cache (recompute decoder over full prefix) to validate correctness; then add KV-cache for speed using a prefill + decode structure.
- Execution: use `pixi run python` for all scripts and harnesses.

## Current decisions (for initial port)

- Checkpoint: `charsiu/g2p_multilingual_byT5_tiny_8_layers_100`.
- Batch size: 1.
- Precision: fp32 first (to simplify numerical validation).
- Scope: TVM + Python harness only (no mobile sample app in this repo).
- Decode strategy: greedy (beam=1) for baseline, evaluate beam search later if needed.
- Length bounds: `max_input_bytes=64`, `max_output_len=128`.

### Reasonable input/output length (TTS use case)

CharsiuG2P is word-level G2P. For TTS, you typically run it on tokenized words (not sentences), so length limits can be fairly small.

Recommended starting bounds:

- `max_input_bytes`: 64 (safe for most words), 96 if you want extra headroom for long Germanic compounds or transliterations.
- `max_output_len`: 128 (IPA is often longer than the graphemes; 2x input is usually safe).

If you want to be more conservative for rare long words, use `max_input_bytes=96` and `max_output_len=160`. Once the harness works, we can compute empirical max lengths from `external/CharsiuG2P/dicts` and tighten these bounds.

### Empirical length stats (from `external/CharsiuG2P/dicts`)

Measured with `pixi run python` over all `dicts/*.tsv` entries, using UTF-8 byte lengths:

| Metric | Count | P50 | P90 | P95 | P99 | P99.9 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `word_bytes` | 7,282,524 | 11 | 18 | 20 | 27 | 38 | 141 | 11.56 |
| `word_prefixed_bytes` (`<lang>:` + word) | 7,282,524 | 18 | 24 | 27 | 33 | 44 | 147 | 18.25 |
| `pron_bytes` | 7,282,524 | 13 | 22 | 26 | 44 | 85 | 370 | 14.52 |

Interpretation:

- The chosen bounds `max_input_bytes=64` and `max_output_len=128` cover well beyond the 99.9th percentile for both inputs and outputs.
- There are long-tail outliers (e.g., `pron_bytes` max 370). If those are important, we can raise bounds or special-case long inputs.

## Mobile deployment focus (iOS + Android)

- Target devices: iPhone/iPad (Metal) and Android (OpenCL/Vulkan or CPU fallback), with Relax VM as the on-device runtime.
- Output format: plan to produce mobile-friendly archives (commonly `.tar`) that bundle the compiled objects and TVM runtime pieces.
- Toolchain: iOS builds must be done on macOS; Android builds can be cross-compiled with NDK.
- Host triples to target: `arm64-apple-ios` (iOS) and `aarch64-linux-android` (Android).
- Goal: ship a minimal runtime footprint (static or slim runtime where possible).

## MLC-LLM-derived hints (useful patterns to copy)

- MLC-LLM compiles for iOS/Android and packages model libraries into `.tar` outputs; mirror that for CharsiuG2P artifacts.
- MLC-LLM links optional `flash_attn` when available; if we later introduce custom attention kernels, we can follow the same pattern (optional link, fallback behavior).
- MLC-LLM’s engine is cross-platform; treat it as a reference for how to structure mobile packaging and runtime dependencies, even if the model/ops differ.

## Concrete build checklist (mobile)

### A. Compile the model (host machine)

1) Export PyTorch model to Relax
   - Use `torch.export.export()` and `tvm.relax.frontend.torch.from_exported_program(...)`.
   - Freeze input shapes using the chosen `max_input_bytes` and `max_output_len`.
2) Build a Relax VM executable
   - Compile with `tvm.compile(mod, target=...)` for each target:
     - iOS: `--device iphone`, host triple `arm64-apple-ios`.
     - Android: `--device android`, host triple `aarch64-linux-android`.
   - Emit a mobile-friendly archive (`.tar`) containing compiled objects and metadata.
3) Package runtime + assets
   - Bundle: compiled archive, TVM runtime libs, tokenizer files, and a minimal config JSON.
   - Keep a “model manifest” with shape limits and dtype.

### B. Build runtime libs

1) TVM runtime (core)
   - Build a slim TVM runtime for each platform (prefer static where feasible).
   - Ensure `relax.VirtualMachine` support is included.
2) Optional: custom kernels / attention
   - If you add custom attention, follow MLC-LLM’s pattern: link optional `flash_attn` when available; otherwise build without it.

### C. iOS integration checklist

1) Build on macOS with Xcode toolchain.
2) Add the compiled `.tar` outputs and tokenizer assets to the app bundle.
3) Link `tvm_runtime` (static or dynamic) into the app target.
4) Implement a thin C++/Objective-C++ bridge:
   - Load the compiled module.
   - Instantiate Relax VM on Metal or CPU.
   - Expose `run_g2p(words, lang)` to Swift.
5) Validate on device:
   - Single-word test from the golden set.
   - Compare output strings against the reference harness.

### D. Android integration checklist

1) Use NDK to build and link `tvm_runtime` into an Android library (`.so`).
2) Add the compiled `.tar` outputs and tokenizer assets into `assets/`.
3) Implement a JNI wrapper:
   - Load the compiled module.
   - Instantiate Relax VM (Vulkan/OpenCL or CPU).
   - Expose `run_g2p(words, lang)` to Kotlin/Java.
4) Validate on device:
   - Single-word test from the golden set.
   - Compare output strings against the reference harness.

## Sample app skeleton (minimal)

### iOS (Swift + Objective-C++)

- Swift UI layer:
  - Text input + language picker.
  - Call `G2PBridge.run(words: [String], lang: String)` and display outputs.
- Objective-C++ bridge (`G2PBridge.mm`):
  - Load model archive and tokenizer assets from the app bundle.
  - Initialize Relax VM and keep it as a singleton.
  - Provide `run(words, lang)` -> `[String]`.

### Android (Kotlin + JNI)

- Kotlin UI layer:
  - Text input + language picker.
  - Call `G2PBridge.run(words, lang)` and display outputs.
- JNI layer (`g2p_jni.cpp`):
  - Load model archive and tokenizer assets from `assets/`.
  - Initialize Relax VM and keep it as a singleton.
  - Provide `run(words, lang)` -> `String[]`.

## Implementation plan (phased, with acceptance criteria)

### 1) Pick target model + constraints

- Decide: checkpoint (`charsiu/g2p_multilingual_byT5_tiny_8_layers_100`), `max_input_bytes`, `max_output_len`, batch size, mobile targets (`iphone`, `android`) plus a dev target (`llvm`).
- Acceptance: frozen "contract" (max lengths, dtype, device targets) you can compile against.

### 2) Build a gold-reference harness (PyTorch)

- Implement `reference_g2p(words, lang) -> phones` using `transformers` (greedy, fixed `max_length`, `num_beams=1`).
- Save a small golden set (10-50 words across 2-3 languages) + expected outputs.
- Acceptance: deterministic outputs (same tokenizer/model/params) and latency baseline.

### 3) Factor the model into TVM-compilable subgraphs

- Create minimal PyTorch wrapper modules with explicit tensor I/O:
  - `EncoderWrapper(input_ids, attention_mask) -> encoder_hidden_states`
  - `DecoderWrapper(decoder_input_ids, encoder_hidden_states, encoder_attention_mask) -> logits`
- Export each via `torch.export.export()` using representative padded shapes.
- Acceptance: exported programs run and match vanilla HF forward numerically (within tolerance).

### 4) Import to Relax and compile a runnable VM module

- Convert via `from_exported_program(...)` (Relax torch frontend), then compile with `tvm.compile(..., target=...)`, run via `relax.VirtualMachine(exec, dev)`.
- Acceptance: end-to-end forward works inside TVM for encoder + decoder, producing logits close to PyTorch.

### 5) MVP generation loop (no KV-cache)

- Implement greedy decode in Python:
  - Start token (T5-style), repeatedly call compiled `DecoderWrapper` on the growing `decoder_input_ids` until EOS or `max_output_len`.
- Acceptance: golden-set pronunciations exactly match (string-level) PyTorch `generate()` for greedy decode, or match token-by-token with fixed EOS/stop rules.

### 6) Performance pass 1: KV-cache design + prefill/decode split

- Add caching to avoid O(L^2) decoder cost:
  - `prefill(decoder_input_ids_prefix, encoder_states) -> (logits, cache)`
  - `decode_step(last_token, cache, encoder_states) -> (logits, cache')`
- Practical constraint: TVM prefers static shapes for best codegen/tuning; model cache as fixed-size `[max_output_len, ...]` + a `cur_pos` index (LLM-style) rather than dynamically growing tensors.
- Acceptance: same outputs as step 5, but with materially better latency for longer outputs.

### 7) Performance pass 2: scheduling/tuning and dtypes

- CPU: rely on standard schedules + fusion; consider `meta_schedule` only if shapes are static enough.
- GPU: apply default GPU pipeline and evaluate fp16/bf16 if acceptable.
- Acceptance: repeatable build script producing mobile artifacts (e.g., `.tar`) and a local dev executable with measured speedups vs PyTorch eager.

### 8) Packaging + ergonomics

- Provide a small Python API mirroring CharsiuG2P usage (tokenized words in, pron strings out), plus a CLI.
- Ship compiled artifacts per target (iOS/Android/dev) and cache by `(model_id, target, max_lens, dtype)`.
- Add mobile packaging docs: what to bundle in an app (compiled archive + tvm runtime libs + tokenizer assets).
- Acceptance: a one-command demo on desktop + a minimal mobile sample app that loads the compiled model and runs one test word.

## Risks / mitigations

- Torch export incompatibilities (T5 attention, relative position bias): mitigate by exporting smaller wrappers, enabling decomposition (`run_ep_decomposition=True`), or falling back to ONNX for forward graphs.
- Cache complexity for encoder-decoder (self-attn + cross-attn): start cacheless; then cache decoder self-attn first; treat cross-attn keys/values as precomputed from encoder states.
- Input prefix exactness (`"<lang>:"` vs `"<lang>: "`): lock this to the exact HF checkpoint behavior via the gold harness and treat it as part of the model contract.

## Beam search: pros and cons (for G2P)

Pros:

- Often improves pronunciation accuracy on ambiguous words or low-resource languages.
- Can reduce rare hallucinations in the decoder (by exploring alternatives).
- Useful if you plan to expose multiple pronunciations (n-best outputs).

Cons:

- Slower and more memory intensive (multiple beams per step).
- Harder to make fully static-shape on-device without over-allocating buffers.
- For these G2P checkpoints, the README suggests greedy decoding is already sufficient.
