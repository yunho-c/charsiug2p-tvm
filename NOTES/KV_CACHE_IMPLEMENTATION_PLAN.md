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
