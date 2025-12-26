# Post-processing notes (English)

This note captures what we learned while mapping CharsiuG2P IPA output into Misaki's English phoneme inventory and evaluating accuracy against Misaki's lexicon.

## Summary of findings

- There is no single deterministic IPA → Misaki mapping across languages; for English, a mostly deterministic mapping is possible but still imperfect.
- The largest error sources are consistent and mostly explainable by inventory/tokenization differences between “traditional” IPA outputs and Misaki’s custom symbols.

## Common failure patterns (from misaki_analysis.csv)

- Dark-L vs plain L: IPA often uses `ɫ` where Misaki uses `l` (largest substitution group). This shows up in words like `chandler`, `puller`, `elijah`.
- Diphthongs compressed into single tokens: Misaki uses `O/I/W/A/Y` while IPA often contains `oʊ/aɪ/aʊ/eɪ/ɔɪ`. This yields frequent mismatches (e.g., `capote`, `skycap`, `bouncers`).
- Rhotic vowels: `ɝ` and `ɚ` in IPA often need to be normalized to `ɜɹ` or `əɹ` to match Misaki output (e.g., `edinburgh`, `misheard`).
- Flaps: IPA tends to keep `t` where Misaki uses `ɾ` in intervocalic contexts (e.g., `skating`, `auditing`).
- Syllabic consonants: IPA `əl/ən/əm` vs Misaki `ᵊl/ᵊn/ᵊm` (e.g., `provenance`, `cudgel`, `abled`).
- Affricate encoding: IPA `tʃ/dʒ` sometimes needs to be normalized to Misaki’s `ʧ/ʤ` tokens.

## Strategy overview

- `espeak`: Minimal mapping (aligned with Misaki’s `from_espeak()` in `external/misaki/EN_PHONES.md`). Intended for espeak outputs with tie markers (`^`). This is the least invasive and serves as a baseline.
- `ipa`: Adds IPA-specific rewrites before the espeak mapping. It normalizes common IPA digraphs (`tʃ`, `dʒ`), diphthongs (`oʊ`, `aɪ`, `aʊ`, `eɪ`, `ɔɪ`), dark-L (`ɫ`), and rhotic vowels (`ɝ`). It also includes heuristics for syllabic consonants. This improves agreement when the input IPA is “traditional” rather than espeak-style.
- `ipa-flap`: Same as `ipa` but also adds a simple flapping heuristic (`t` between vowels → `ɾ`). This is useful when CharsiuG2P outputs `t` for intervocalic flaps but Misaki’s lexicon uses `ɾ`.

## Implementation pointers

- Mapping lives in `python/src/charsiug2p_tvm/post_processing.py`. The `strategy` argument selects the rewrite pipeline.
- The analysis CLI is `python -m charsiug2p_tvm analyze-misaki` with `--strategy` repeatable or comma-separated.
- The output CSV includes per-strategy columns so we can compare exact-match/CER trends side-by-side.

## Plan (updated)

- First implement failure-mode classification with per-mode metrics (so each heuristic can be validated independently).
- Then expand sample size or stratify samples by failure mode for more robust evaluation.

## Implementation status

- Failure-mode classification + per-mode metrics: implemented in `python/src/charsiug2p_tvm/misaki_analysis.py` and surfaced in the CLI output (`analyze-misaki`).
- Strategy comparison remains available via `--strategy` (repeatable or comma-separated).

## Suggested next experiments

- Run `analyze-misaki` with `--strategy espeak,ipa,ipa-flap` on a larger sample to see if the additional IPA rewrites help consistently.
- Add optional toggles for individual IPA rewrites (dark-L, diphthongs, syllabic consonants) to isolate which changes help or hurt.
