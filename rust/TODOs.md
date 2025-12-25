# Rust TODOs

- Add KV-cache support in `g2p_pipeline` using `decoder_prefill` and `decoder_step` artifacts.
- Extend artifact discovery to read a manifest (checksum, shape bounds, dtype, target metadata) and validate inputs.
- Add Relax VM runtime support in `g2p_tvm` once `tvm-ffi` exposes the necessary APIs, or add a small C wrapper if needed.
- Add SentencePiece-based tokenizer fallback when `tokenizer.json` is unavailable.
- Add unit tests for `ArtifactResolver`, including env override and fallback path coverage.
- Add CLI options like `--print-paths` and `--dry-run` to debug artifact resolution.
- Add batch-size selection logic that picks the smallest compiled batch size that fits each chunk.
- Define a mobile asset loader interface for per-platform storage paths and package layout.
