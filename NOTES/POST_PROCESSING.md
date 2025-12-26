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
- `ipa-vowel`: Extends `ipa` with vowel tuning. It promotes stressed schwas to `ʌ`, demotes unstressed `ɜɹ` to `əɹ`, maps unstressed `ɪd/ɪz` to `ᵻd/ᵻz`, and adds an explicit `əɫ → ᵊl` rewrite while avoiding blanket `ən/əm → ᵊn/ᵊm` conversions (favoring syllabic markers instead).
- `ipa-flap-vowel`: Combines `ipa-vowel` with the flapping heuristic.
- `ipa-vowel-stress`: Adds a stress-normalization pass on top of `ipa-vowel`. It demotes extra primary stresses after the first, and drops secondary stress on reduced vowels (`ɪ/ᵻ/ə/ᵊ`) when a primary stress already exists.
- `ipa-flap-vowel-stress`: Combines `ipa-vowel-stress` with the flapping heuristic.
- `ipa-vowel-prefix` (experimental): Adds prefix-aware stress rules on top of `ipa-vowel`. If a prefixed word has exactly one primary and one secondary stress but the primary appears before the secondary, the stresses are swapped. For short prefixes (`re/un/in/im/il/ir`), an initial `ˌ` is dropped when there is only one secondary stress marker. Current results show this underperforms, so it is not recommended for default comparisons.
- `ipa-flap-vowel-prefix` (experimental): Combines `ipa-vowel-prefix` with the flapping heuristic. Also underperforms relative to `ipa-flap-vowel`.
- `ipa-vowel-syllabic` (experimental): Uses the same vowel tuning as `ipa-vowel`, but swaps in a context-sensitive `ə + l/n → ᵊl/ᵊn` rule that only fires after consonants and before consonants. This avoids over-converting sequences like `əlo`.
- `ipa-flap-vowel-syllabic` (experimental): Combines `ipa-vowel-syllabic` with the flapping heuristic.
- `ipa-vowel-reduced`: Adds a conservative pretonic reduction rule (`ɪ → ə`) in consonant contexts when a primary stress appears later in the word.
- `ipa-flap-vowel-reduced`: Combines `ipa-vowel-reduced` with the flapping heuristic. This is currently the best-performing strategy (English) and is recommended for comparisons.

## Implementation pointers

- Mapping lives in `python/src/charsiug2p_tvm/post_processing.py`. The `strategy` argument selects the rewrite pipeline.
- The analysis CLI is `python -m charsiug2p_tvm analyze-misaki` with `--strategy` repeatable or comma-separated.
- The output CSV includes per-strategy columns so we can compare exact-match/CER trends side-by-side.

## Plan (updated)

- First implement failure-mode classification with per-mode metrics (so each heuristic can be validated independently).
- Then expand sample size or stratify samples by failure mode for more robust evaluation.

## Implementation status

- Failure-mode classification + per-mode metrics: implemented in `python/src/charsiug2p_tvm/misaki_analysis.py` and surfaced in the CLI output (`analyze-misaki`).
- Coverage and primary-mode summaries are also available to separate prevalence from per-mode accuracy.
- Strategy comparison remains available via `--strategy` (repeatable or comma-separated).

## Suggested next experiments

- Run `analyze-misaki` with `--strategy espeak,ipa,ipa-flap` on a larger sample to see if the additional IPA rewrites help consistently.
- Add optional toggles for individual IPA rewrites (dark-L, diphthongs, syllabic consonants) to isolate which changes help or hurt.
- Use `--mode-sub-report --mode-sub-modes syllabic_schwa,reduced_vowel` to inspect the dominant substitution pairs and guide heuristics.

## Context and implementation plan (diff-based mode tagging)

### Why change mode tagging

Current failure modes are presence-based: if a string contains `ə` or `ɜ`, it gets tagged `reduced_vowel` even when that token is not part of the mismatch. This inflates mode counts and makes the primary-mode table less meaningful. Diff-based tagging will only assign a mode when the mismatch actually involves a relevant substitution or insertion/deletion.

### Plan overview

1) Tokenize phoneme strings into units before diffing.
   - Keep stress markers (`ˈ`, `ˌ`) as standalone tokens.
   - Collapse multi-character tokens so the diff is more faithful to phoneme units.
   - Suggested collapse set (longest-first):
     - Diphthongs: `aɪ`, `aʊ`, `eɪ`, `oʊ`, `ɔɪ`, `əʊ`
     - Affricates: `tʃ`, `dʒ`
     - Rhotic combos: `ɜɹ`, `əɹ`
     - Syllabic combos: `ᵊl`, `ᵊn`, `ᵊm` (or keep `ᵊ` + consonant as separate tokens if simpler)

2) Compute edit ops on token sequences.
   - Implement a Levenshtein backtrace that yields `sub(old, new)`, `ins(new)`, `del(old)` at the token level.
   - Reuse the existing `_edit_distance` pattern but adapt it to token arrays.

3) Map edit ops to modes (examples).
   - `stress_only`: all ops are stress insertions/deletions/moves.
   - `stress_mismatch`: any op involving `ˈ` or `ˌ`.
   - `diphthong_token`: ops like `aɪ <-> I`, `aʊ <-> W`, `oʊ <-> O`, `eɪ <-> A`, `ɔɪ <-> Y`, `əʊ <-> Q`.
   - `affricate_token`: ops like `tʃ <-> ʧ`, `dʒ <-> ʤ`.
   - `rhotic_vowel`: ops involving `ɝ`, `ɚ`, `ɜɹ`, `əɹ`.
   - `syllabic_schwa`: ops involving `ᵊ` or `ᵊl/ᵊn/ᵊm` vs `əl/ən/əm`.
   - `flap`: `t <-> ɾ` substitutions or `ɾ` insert/delete.
   - `reduced_vowel`: ops among `ə/ɪ/ʌ/ɜ/ᵻ/i` (but exclude those already tagged by rhotic or syllabic rules).
   - Fallback: `other`.

4) Replace the current presence-based `classify_failure_modes(ref, hyp)` with op-based tagging.
   - Keep the primary-mode priority list, but apply it to the op-derived tags.
   - Continue to tag `match` for exact matches so coverage tables still include that category.

5) Validate and compare.
   - Expect lower per-mode coverage (more realistic prevalence).
   - Expect primary-mode distribution to shift away from stress dominance.
   - Compare pre/post diff-based tagging on the same dataset to confirm interpretability improvements.

### Optional enhancements

- Add a CLI switch to choose presence-based vs diff-based tagging for quick A/B comparison.
- Export per-op substitution counts (top substitutions) for targeted heuristic tuning.
