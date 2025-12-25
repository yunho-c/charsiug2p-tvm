# Rust TODOs

- Extend artifact discovery to read a manifest (checksum, shape bounds, dtype, target metadata) and validate inputs.
- Add SentencePiece-based tokenizer fallback when `tokenizer.json` is unavailable.
- Add unit tests for `ArtifactResolver`, including env override and fallback path coverage.
- Add CLI options like `--print-paths` and `--dry-run` to debug artifact resolution.
- Define a mobile asset loader interface for per-platform storage paths and package layout.
