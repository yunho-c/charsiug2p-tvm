# Rust post-processing port (ipa-flap-vowel-reduced)

## Goal

Port the Python `ipa-flap-vowel-reduced` strategy into the Rust crates so the Rust CLI and Flutter FFI can optionally emit Misaki-compatible tokens. Keep the default Rust output unchanged unless a post-process strategy is explicitly selected.

## Recommended design

- Add a reusable post-processing module (suggested path: `rust/g2p_pipeline/src/post_processing.rs`).
- Expose an enum and entrypoint:
  - `enum PostProcessStrategy { IpaFlapVowelReduced, ... }`
  - `fn apply(strategy: PostProcessStrategy, ipa: &str, british: bool, word: Option<&str>) -> String`
- Add a config field on the pipeline instead of a new method:
  - `PipelineConfig { post_process: Option<PostProcessStrategy>, post_process_british: bool }`
  - Default `post_process = None` to preserve current behavior.
- Call the post-processing function after decoding in `G2pPipeline::run` (and in the system-lib path).
- Thread config choices through `g2p_cli` and Flutter FFI (CLI flag, FFI config field) so both entrypoints share the same logic.

## Port checklist for `ipa-flap-vowel-reduced`

Mirror the Python behavior in `python/src/charsiug2p_tvm/post_processing.py`:

1) IPA rewrites (vowel-tuned table):
   - `tʃ → t^ʃ`, `dʒ → d^ʒ`
   - `oʊ → o^ʊ`, `əʊ → ə^ʊ`, `aɪ → a^ɪ`, `aʊ → a^ʊ`, `eɪ → e^ɪ`, `ɔɪ → ɔ^ɪ`
   - `ɝ → ɜɹ`, `ɫ → l`, `əl/əɫ → ə^l`
   - Apply longest-match first.

2) From-espeak replacements (longest-match first):
   - `a^ɪ → I`, `a^ʊ → W`, `e/e^ɪ → A`, `ɔ^ɪ → Y`
   - `t^ʃ → ʧ`, `d^ʒ → ʤ`, `r → ɹ`, `ɚ → əɹ`
   - (Other entries exist in Python but do not appear in CharsiuG2P IPA output today; keep table parity.)

3) Syllabic swap:
   - Replace U+0361 tie bar with `^`, then convert `(\S)̩ → ᵊ\1` (U+0329).
   - Remove any remaining U+0329.

4) US-specific replacements:
   - `o^ʊ → O`, `ɜːɹ → ɜɹ`, `ɜː → ɜɹ`, `ɪə → iə`, and remove `ː`.
   - Remove `^`.

5) Vowel tuning:
   - If stressed, `ə → ʌ`; if stressed, `ᵊ → ə`.
   - `ɜɹ` vs `əɹ`: choose `ɜɹ` when stressed, else `əɹ`.
   - `ɪd/ɪz` becomes `ᵻd/ᵻz` when unstressed.
   - Stress markers are reordered only when `ipa-vowel-stress` is requested; do not apply here.

6) Reduced-vowel tuning:
   - `ɪ → ə` when unstressed, preceded by a consonant, followed by a consonant, and a primary stress appears later in the word.
   - Exclude cases where the next token is `ˈ/ˌ` or `ŋ`.

7) Flapping:
   - Convert `t → ɾ` when between vowels, skipping stress markers.

## Implementation plan (config field)

1) Add a new Rust module and a small unit test harness in `rust/g2p_pipeline`.
2) Define `PostProcessStrategy` and parse from CLI/FFI config.
3) Extend `PipelineConfig` with `post_process` and `post_process_british`.
4) Apply post-processing in `G2pPipeline::run` after decoding each output.
5) Add CLI flag `--post-process` (default `none`) and `--post-process-british` if needed.
6) Update Flutter FFI config to include the strategy (string or enum).
7) Add snapshot tests for key transformations: diphthongs, `ɪd/ɪz`, `ɝ/ɚ`, `t` flapping, `ə→ʌ`.

## Notes

- Use `char` iteration and explicit Unicode handling (no byte offsets) due to IPA symbols.
- Keep the output unchanged when `post_process` is unset to avoid behavior regressions.
