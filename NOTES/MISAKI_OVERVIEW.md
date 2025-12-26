# misaki submodule overview

The repo now includes `external/misaki` as a git submodule. Misaki is a G2P engine for Kokoro models with per-language Python entrypoints and optional dependencies declared in `external/misaki/pyproject.toml` (extras: en, ja, ko, zh, vi, he).

## Core structure

- `misaki/token.py` defines `MToken` with optional phoneme fields and an `Underscore` attribute container; most tokenizers fill these and return `(phoneme_string, tokens)`.
- `misaki/espeak.py` wraps phonemizer/espeak-ng and normalizes espeak symbols into Misaki's phoneme inventory. Used as an English fallback (`EspeakFallback`) or general-purpose G2P (`EspeakG2P`).
- `misaki/en.py` is the English pipeline: spaCy tokenization/tagging, lexicon lookup from bundled US/GB gold/silver dictionaries in `misaki/data`, custom sub-tokenization and morphological rules, and fallback either via a BART grapheme-to-phoneme model or espeak. Link-style inline overrides like `[word](/phonemes/)` are supported.
- `misaki/ja.py` provides `JAG2P` with two modes: `cutlet` (romaji) or `pyopenjtalk`. It maps kana to a custom phoneme set and emits a combined phoneme+pitch string for the pyopenjtalk path.
- `misaki/ko.py` uses `g2pkc.G2p` (copied from g2pK via 5Hyeons) for Korean G2P.
- `misaki/zh.py` has a legacy path (jieba + pypinyin + `transcription.py` pinyin-to-IPA) and a v1.1 frontend (`zh_frontend.py`) adapted from PaddleSpeech with tone sandhi and erhua handling. English substrings can be routed to the English callable.
- `misaki/vi.py` is a Vietnamese pipeline with a large custom IPA mapping, optional English fallback, and a text-normalization pass via `vi_cleaner`. It also supports link-style phoneme overrides.
- `misaki/he.py` is a thin wrapper over the `mishkal` Hebrew phonemizer.

## Subpackages and data

- `misaki/g2pkc/` is a copy of g2pK with Mecab-based annotation, Korean rules (`rules.txt`, `table.csv`), English word conversion via CMUdict, and number handling.
- `misaki/zh_normalization/` is copied from PaddleSpeech for Chinese NSW normalization (dates, times, numbers, phone numbers, measures, etc.).
- `misaki/vi_cleaner/` is adapted from Vietnamese text normalization; it expands abbreviations/acronyms, numbers, dates, measurement units, and currency before the G2P stage.
- `misaki/data/` contains dictionaries and language resources (US/GB lexicons, Vietnamese mappings, Japanese word list).

## Notable behaviors

- English uses lexicon-first lookup with morphology-aware fallbacks and optional transformer-based G2P; phonemes can be modified by inline markup, stress flags, and numeric parsing logic.
- Japanese `pyopenjtalk` path produces pitch markers aligned to phoneme strings; cutlet path outputs romaji-like phonemes.
- Chinese v1.1 path uses tone sandhi and a phoneme mapping table derived from PaddleSpeech; legacy path uses pinyin-to-IPA conversion.
- Vietnamese path applies normalization, tokenization, then G2P with a custom onset/nucleus/coda/tone mapping; it can fallback to English for non-VI substrings.

## Entry points

- `external/misaki/examples/usage.py` demonstrates English usage with inline phoneme overrides.
- `external/misaki/README.md` documents per-language dependencies and usage patterns.

## Empirical mapping analysis

- CLI: `python -m charsiug2p_tvm analyze-misaki` compares CharsiuG2P IPA (dataset or model) against Misaki's English lexicon using the espeak-style mapping plus optional stress reordering.
