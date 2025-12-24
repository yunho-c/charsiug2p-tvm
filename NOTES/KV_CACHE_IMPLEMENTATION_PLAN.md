# KV-Cache Implementation Plan (T5/ByT5)

This plan describes how to add KV-cache to the TVM path while keeping static shapes for compilation.

## Goals

- Add prefill + decode-step entrypoints that use cached decoder self-attn.
- Keep shapes static for TVM (fixed max lengths, fixed batch size).
- Maintain correctness vs the reference `transformers` path.

## Model assumptions (T5/ByT5)

- Decoder self-attn cache shapes per layer:
  - K: [batch, num_heads, seq_len, head_dim]
  - V: [batch, num_heads, seq_len, head_dim]
- `head_dim` is `config.d_kv` (T5 uses separate `d_kv`).
- Total decoder layers: `config.num_decoder_layers`.

## Cache layout choice

To make export/compile simpler, pack caches into dense tensors:

- `cache_k`: [num_layers, batch, num_heads, max_output_len, head_dim]
- `cache_v`: [num_layers, batch, num_heads, max_output_len, head_dim]
- `cur_pos`: scalar int (current decode position)

This avoids Python lists of tuples in the exported graph and keeps shapes static.

## Export pipeline changes (tvm_compile.py)

1) Read config values from the checkpoint
   - `num_layers`, `num_heads`, `d_kv`, `d_model`

2) Add two wrapper modules for torch.export

- Prefill wrapper
  - Inputs: `decoder_input_ids_prefix`, `encoder_hidden_states`, `encoder_attention_mask`
  - Outputs: `logits_prefix`, `cache_k`, `cache_v`, `cur_pos`
  - Behavior:
    - Run decoder with `use_cache=True` on the prefix
    - Write the returned past_key_values into `cache_k/cache_v`
    - Set `cur_pos = prefix_length`

- Decode wrapper
  - Inputs: `last_token_id`, `encoder_hidden_states`, `encoder_attention_mask`, `cache_k`, `cache_v`, `cur_pos`
  - Outputs: `logits_step`, `cache_k`, `cache_v`, `cur_pos+1`
  - Behavior:
    - Build `past_key_values` from `cache_k/cache_v` slices
    - Run decoder for a single step with `use_cache=True`
    - Update cache at `cur_pos`

3) Export both wrappers via `torch.export.export`

- Keep shapes fixed:
  - `decoder_input_ids_prefix`: [batch, max_output_len]
  - `last_token_id`: [batch, 1]
  - `cache_k/cache_v`: fixed max length

4) Build Relax artifacts

- Compile to separate modules:
  - `decoder_prefill` and `decoder_step`
- Keep encoder module as-is.

## Runtime changes (tvm_runtime.py)

1) Add a `TvmG2PWithCache` runtime

- Load encoder + prefill + step artifacts.
- Pre-allocate `cache_k/cache_v` on device (tvm runtime tensors).
- Track `cur_pos` on host and pass to step call.

2) Greedy decode loop

- Prefill once with the entire prefix (decoder_input_ids containing just BOS/pad at position 0).
- Use step function to generate tokens one-by-one, updating cache each step.

3) Masking

- Use `encoder_attention_mask` as currently.
- Ensure decoder uses causal masking internally (T5 does this).

## Testing/validation

- Run `verify` on a small set and compare to reference.
- Confirm logits match within tolerance on a single step.
- Ensure output strings match for 10-100 samples.

## Known challenges

- `torch.export` support for cache updates and tuple outputs; packing caches into fixed tensors should help.
- Past key/value ordering and layout must match T5 expectations.
- Static `max_output_len` means we must guard against overflow (stop at max length).

## Suggested rollout

1) Add cache tensors to the wrapper but ignore them (smoke export).
2) Wire prefill only and validate logits equivalence.
3) Add step function and update runtime loop.
4) Turn on `verify` and check exact match + CER.

## Implementation status (current)

- `export_torch_model_with_cache` now exports a prefill module that returns fixed-size caches and a step module that accepts fixed-size caches plus a `cur_pos` input.
- Prefill uses `Cache(layer_class_to_replicate=ExportDynamicLayer)` with a custom `ExportEncoderDecoderCache` to avoid `torch.export` guards, then sets `cur_pos` from `cache_position[-1] + 1`.
- Decode step uses fixed-length `past_key_values` (length = `max_output_len`) and a `cur_pos` scalar to update caches; `cache_position` is built from a fixed-length tensor plus `cur_pos` to avoid symbolic-length `arange`.
- `ExportDynamicLayer` initializes 4D empty tensors for K/V and updates via `torch.cat`, then trims to `max_cache_len` to keep shapes fixed and avoid `index_put`.
- Runtime (`tvm_runtime.py`) passes `cur_pos` into the step VM call and reuses returned `cur_pos` for the next iteration.

## Known working compile command

- `pixi run python -m charsiug2p_tvm compile --kv-cache --target llvm`
- Artifacts end up under `python/dist/tvm/charsiu_g2p_multilingual_byT5_tiny_8_layers_100/b1_in64_out128/llvm` (encoder, decoder_prefill, decoder_step).

## Export failure notes (what broke and what fixed it)

- `DynamicCache` starts with 1D empty tensors; `torch.cat` then mixes rank-1 and rank-4 → Relax `concat` error. Fix: custom layer with 4D empty tensors.
- `StaticCache` updates via `index_copy_`/`index_put` → Relax `index_put` expects full-dim indices. Fix: replace in-place update with `cat` + slice.
- `EncoderDecoderCache.__init__` calls `get_seq_length()` and converts it to `bool` during export → data-dependent guard error. Fix: minimal subclass that avoids the bool guard and stores `cur_pos` explicitly.
- `torch.export` specialized `cache_len` to 1 when using dynamic cache length → constraints violation. Fix: use fixed-length caches and carry `cur_pos` as an explicit input.
- `torch.arange(cur_pos, cur_pos + 1)` created a symbolic-length tensor → guard error. Fix: build `cache_position` from a fixed-size tensor and add `cur_pos`.

## Beginner-friendly explanation (why these changes compile)

TVM’s exporter likes static shapes and simple tensor ops. The original KV-cache path was too dynamic:

- It built caches that *grow* over time.
- It used in-place index updates (`index_put`) that TVM doesn’t lower well.
- It created tensors whose length depended on runtime values (`cur_pos`), which triggers export guards.

To make it compile, we made the cache **fixed-size** and moved all “dynamic” behavior into plain tensor math:

- **Fixed-size cache tensors**: we always allocate `[num_layers, batch, num_heads, max_output_len, head_dim]`.
- **Append via concat + slice**: instead of `index_put`, we `cat` the new token and slice to `max_output_len`.
- **Track position explicitly**: we carry a small `cur_pos` scalar between steps instead of relying on dynamic shapes.
- **Avoid symbolic lengths**: `cache_position` is built from a fixed-length tensor plus `cur_pos`, not `arange` of symbolic size.

These changes keep the graph static and predictable while still behaving like a KV-cache at runtime.
