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

## Implementation plan (phased, with acceptance criteria)

### 1) Pick target model + constraints

- Decide: checkpoint (e.g., `charsiu/g2p_multilingual_byT5_tiny_16_layers_100`), `max_input_bytes`, `max_output_len`, batch size, targets (`llvm`, `cuda`).
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
- Acceptance: repeatable build script producing a `.so` (or VM executable) with measured speedups vs PyTorch eager.

### 8) Packaging + ergonomics

- Provide a small Python API mirroring CharsiuG2P usage (tokenized words in, pron strings out), plus a CLI.
- Ship compiled artifacts per target (CPU/GPU) and cache by `(model_id, target, max_lens, dtype)`.
- Acceptance: `pip install -e python` + one-command demo on a clean machine with only TVM runtime + your package.

## Risks / mitigations

- Torch export incompatibilities (T5 attention, relative position bias): mitigate by exporting smaller wrappers, enabling decomposition (`run_ep_decomposition=True`), or falling back to ONNX for forward graphs.
- Cache complexity for encoder-decoder (self-attn + cross-attn): start cacheless; then cache decoder self-attn first; treat cross-attn keys/values as precomputed from encoder states.
- Input prefix exactness (`"<lang>:"` vs `"<lang>: "`): lock this to the exact HF checkpoint behavior via the gold harness and treat it as part of the model contract.
