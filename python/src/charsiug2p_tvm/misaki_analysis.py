from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import csv
import json
import random

from charsiug2p_tvm.harness import reference_g2p
from charsiug2p_tvm.post_processing import (
    STRESS_PREFIXES,
    espeak_ipa_to_misaki,
    fix_stress_placement,
    normalize_strategy,
)

_STRESS_TRANSLATION = str.maketrans("", "", "ˈˌ")


@dataclass(frozen=True)
class AnalysisSample:
    word: str
    misaki: str
    charsiu_ipa: str
    mapped_by_strategy: dict[str, str]
    mapped_stress_by_strategy: dict[str, str]


@dataclass(frozen=True)
class MappingMetrics:
    name: str
    exact_match: int
    exact_match_rate: float
    cer: float


@dataclass(frozen=True)
class FailureModeMetrics:
    name: str
    total: int
    exact_match: int
    exact_match_rate: float
    cer: float


@dataclass(frozen=True)
class FailureModeCoverage:
    name: str
    total: int
    rate: float


@dataclass(frozen=True)
class AnalysisReport:
    total: int
    dataset_words: int
    lexicon_words: int
    strategies: list[str]
    metrics: list[MappingMetrics]
    mode_metrics_by_strategy: dict[str, list[FailureModeMetrics]]
    mode_coverage_by_strategy: dict[str, list[FailureModeCoverage]]
    primary_mode_metrics_by_strategy: dict[str, list[FailureModeMetrics]]
    samples: list[AnalysisSample] | None


@dataclass(frozen=True)
class StressPrefixMetrics:
    prefix: str
    any_total: int
    primary_total: int
    swap_candidates: int
    extra_initial_secondary: int


def _grow_dictionary(entries: dict[str, object]) -> dict[str, object]:
    expanded: dict[str, object] = {}
    for key, value in entries.items():
        if len(key) < 2:
            continue
        if key == key.lower():
            if key != key.capitalize():
                expanded[key.capitalize()] = value
        elif key == key.lower().capitalize():
            expanded[key.lower()] = value
    expanded.update(entries)
    return expanded


def _load_lexicon_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_misaki_lexicon(misaki_root: Path, *, british: bool) -> dict[str, str]:
    data_root = misaki_root / "misaki" / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"Misaki data dir not found: {data_root}")

    suffix = "gb" if british else "us"
    gold_path = data_root / f"{suffix}_gold.json"
    silver_path = data_root / f"{suffix}_silver.json"
    gold = _grow_dictionary(_load_lexicon_json(gold_path))
    silver = _grow_dictionary(_load_lexicon_json(silver_path))

    merged: dict[str, object] = {**silver, **gold}
    normalized: dict[str, str] = {}
    for word, value in merged.items():
        if isinstance(value, str):
            normalized[word] = value
        elif isinstance(value, dict):
            default = value.get("DEFAULT")
            if isinstance(default, str):
                normalized[word] = default
    return normalized


def load_charsiu_tsv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"CharsiuG2P TSV not found: {path}")
    entries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word, pron = parts[0], parts[1]
            if "," in pron:
                pron = pron.split(",", 1)[0]
            if word not in entries:
                entries[word] = pron
    return entries


def _iter_batches(values: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def _strip_stress(text: str) -> str:
    return text.translate(_STRESS_TRANSLATION)


_TOKEN_UNITS = [
    "ᵊl",
    "ᵊn",
    "ᵊm",
    "əl",
    "ən",
    "əm",
    "aɪ",
    "aʊ",
    "eɪ",
    "oʊ",
    "ɔɪ",
    "əʊ",
    "tʃ",
    "dʒ",
    "ɜɹ",
    "əɹ",
]


def _tokenize_phonemes(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        if text[i] in {"ˈ", "ˌ"}:
            tokens.append(text[i])
            i += 1
            continue
        matched = False
        for unit in _TOKEN_UNITS:
            if text.startswith(unit, i):
                tokens.append(unit)
                i += len(unit)
                matched = True
                break
        if matched:
            continue
        tokens.append(text[i])
        i += 1
    return tokens


def _edit_ops(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> list[tuple[str, str, str]]:
    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops: list[tuple[str, str, str]] = []
    i = len(ref_tokens)
    j = len(hyp_tokens)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref_tokens[i - 1] != hyp_tokens[j - 1]):
            if ref_tokens[i - 1] != hyp_tokens[j - 1]:
                ops.append(("sub", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref_tokens[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp_tokens[j - 1]))
            j -= 1
    return ops


def classify_failure_modes(ref: str, hyp: str) -> set[str]:
    if ref == hyp:
        return set()

    tags: set[str] = set()
    ops = _edit_ops(_tokenize_phonemes(ref), _tokenize_phonemes(hyp))
    if not ops:
        return tags

    stress_tokens = {"ˈ", "ˌ"}
    stress_ops = []
    for op, old, new in ops:
        if op == "sub" and old in stress_tokens and new in stress_tokens:
            stress_ops.append(True)
        elif op == "del" and old in stress_tokens:
            stress_ops.append(True)
        elif op == "ins" and new in stress_tokens:
            stress_ops.append(True)
    if stress_ops:
        tags.add("stress_mismatch")
        if len(stress_ops) == len(ops):
            tags.add("stress_only")

    ipa_diph = {"aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "əʊ"}
    misaki_diph = {"A", "I", "W", "O", "Y", "Q"}
    affricate_ipa = {"tʃ", "dʒ"}
    affricate_misaki = {"ʧ", "ʤ"}
    rhotic_tokens = {"ɝ", "ɚ", "ɜɹ", "əɹ"}
    syllabic_tokens = {"ᵊ", "ᵊl", "ᵊn", "ᵊm", "əl", "ən", "əm"}
    reduced_vowels = {"ə", "ɜ", "ɪ", "ʌ", "ᵻ", "i"}

    for op, old, new in ops:
        if op == "sub" and ((old in ipa_diph and new in misaki_diph) or (old in misaki_diph and new in ipa_diph)):
            tags.add("diphthong_token")
        if op == "sub" and (
            (old in affricate_ipa and new in affricate_misaki) or (old in affricate_misaki and new in affricate_ipa)
        ):
            tags.add("affricate_token")
        if old in rhotic_tokens or new in rhotic_tokens:
            tags.add("rhotic_vowel")
        if old in syllabic_tokens or new in syllabic_tokens:
            tags.add("syllabic_schwa")
        if (old == "ɾ" or new == "ɾ") and (old == "t" or new == "t" or op != "sub"):
            tags.add("flap")

    if "rhotic_vowel" not in tags and "syllabic_schwa" not in tags:
        for op, old, new in ops:
            if old in reduced_vowels or new in reduced_vowels:
                tags.add("reduced_vowel")
                break

    if not tags:
        tags.add("other")
    return tags


_MODE_PRIORITY = [
    "stress_only",
    "stress_mismatch",
    "syllabic_schwa",
    "rhotic_vowel",
    "diphthong_token",
    "affricate_token",
    "flap",
    "reduced_vowel",
    "other",
]


def pick_primary_mode(tags: set[str]) -> str:
    if not tags:
        return "none"
    for mode in _MODE_PRIORITY:
        if mode in tags:
            return mode
    return "other"


def _match_stress_prefix(word: str) -> str | None:
    lowered = word.lower()
    for prefix in STRESS_PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            return prefix
    return None


def _stress_signature(text: str) -> list[str]:
    return [char for char in text if char in {"ˈ", "ˌ"}]


def _first_stress(text: str) -> str | None:
    for char in text:
        if char in {"ˈ", "ˌ"}:
            return char
    return None


def _resolve_words(
    scope: str,
    dataset: dict[str, str],
    lexicon: dict[str, str],
) -> list[str]:
    if scope == "intersection":
        if not dataset:
            raise ValueError("Charsiu dataset is required for scope=intersection.")
        return [word for word in dataset.keys() if word in lexicon]
    if scope == "misaki":
        return list(lexicon.keys())
    raise ValueError(f"Unknown scope: {scope}")


def analyze_misaki_english(
    *,
    charsiu_path: Path,
    misaki_root: Path,
    lang: str,
    british: bool,
    source: str,
    scope: str,
    strategies: Sequence[str],
    limit: int | None,
    shuffle: bool,
    seed: int | None,
    device: str,
    batch_size: int,
    include_samples: bool,
) -> AnalysisReport:
    normalized_strategies = [normalize_strategy(strategy) for strategy in strategies]
    if not normalized_strategies:
        raise ValueError("At least one strategy is required.")
    lexicon = load_misaki_lexicon(misaki_root, british=british)
    dataset = load_charsiu_tsv(charsiu_path)

    words = _resolve_words(scope, dataset, lexicon)
    if source == "dataset" and scope == "misaki":
        raise ValueError("source=dataset is only supported with scope=intersection.")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(words)
    if limit is not None:
        words = words[:limit]

    charsiu_outputs: list[str] = []
    if source == "dataset":
        charsiu_outputs = [dataset[word] for word in words]
    elif source == "model":
        for batch in _iter_batches(words, batch_size):
            charsiu_outputs.extend(
                reference_g2p(
                    batch,
                    lang,
                    device=device,
                )
            )
    else:
        raise ValueError(f"Unknown source: {source}")

    mapped_outputs: dict[str, list[str]] = {strategy: [] for strategy in normalized_strategies}
    mapped_stress_outputs: dict[str, list[str]] = {strategy: [] for strategy in normalized_strategies}
    misaki_outputs: list[str] = []
    samples: list[AnalysisSample] = []

    for word, ipa in zip(words, charsiu_outputs):
        misaki = lexicon[word]
        mapped_by_strategy: dict[str, str] = {}
        mapped_stress_by_strategy: dict[str, str] = {}
        for strategy in normalized_strategies:
            mapped = espeak_ipa_to_misaki(ipa, british=british, strategy=strategy, word=word)
            mapped_stress = fix_stress_placement(mapped)
            mapped_by_strategy[strategy] = mapped
            mapped_stress_by_strategy[strategy] = mapped_stress
            mapped_outputs[strategy].append(mapped)
            mapped_stress_outputs[strategy].append(mapped_stress)
        misaki_outputs.append(misaki)
        if include_samples:
            samples.append(
                AnalysisSample(
                    word=word,
                    misaki=misaki,
                    charsiu_ipa=ipa,
                    mapped_by_strategy=mapped_by_strategy,
                    mapped_stress_by_strategy=mapped_stress_by_strategy,
                )
            )

    total_samples = len(misaki_outputs)
    metrics: list[MappingMetrics] = []
    for strategy in normalized_strategies:
        for suffix, outputs in (
            ("", mapped_outputs[strategy]),
            ("+stress", mapped_stress_outputs[strategy]),
        ):
            name = f"{strategy}{suffix}"
            exact = sum(1 for ref, hyp in zip(misaki_outputs, outputs) if ref == hyp)
            cer = (
                sum(_cer(ref, hyp) for ref, hyp in zip(misaki_outputs, outputs)) / total_samples
                if total_samples
                else 0.0
            )
            metrics.append(
                MappingMetrics(
                    name=name,
                    exact_match=exact,
                    exact_match_rate=exact / total_samples if total_samples else 0.0,
                    cer=cer,
                )
            )

    mode_metrics_by_strategy: dict[str, list[FailureModeMetrics]] = {}
    mode_coverage_by_strategy: dict[str, list[FailureModeCoverage]] = {}
    primary_mode_metrics_by_strategy: dict[str, list[FailureModeMetrics]] = {}
    for strategy in normalized_strategies:
        mode_counts: dict[str, dict[str, float]] = {}
        coverage_counts: dict[str, float] = {}
        primary_counts: dict[str, dict[str, float]] = {}

        for ref, hyp in zip(misaki_outputs, mapped_stress_outputs[strategy]):
            tags = classify_failure_modes(ref, hyp)
            if not tags:
                tags = {"match"}
            primary = pick_primary_mode(tags) if tags != {"match"} else "match"
            cer = _cer(ref, hyp)

            for tag in tags:
                stats = mode_counts.setdefault(tag, {"total": 0.0, "exact": 0.0, "cer": 0.0})
                stats["total"] += 1.0
                stats["exact"] += 1.0 if ref == hyp else 0.0
                stats["cer"] += cer

                coverage_counts[tag] = coverage_counts.get(tag, 0.0) + 1.0

            primary_stats = primary_counts.setdefault(primary, {"total": 0.0, "exact": 0.0, "cer": 0.0})
            primary_stats["total"] += 1.0
            primary_stats["exact"] += 1.0 if ref == hyp else 0.0
            primary_stats["cer"] += cer

        mode_metrics: list[FailureModeMetrics] = []
        for tag, stats in mode_counts.items():
            tag_total = int(stats["total"])
            exact = int(stats["exact"])
            cer = stats["cer"] / tag_total if tag_total else 0.0
            mode_metrics.append(
                FailureModeMetrics(
                    name=tag,
                    total=tag_total,
                    exact_match=exact,
                    exact_match_rate=exact / tag_total if tag_total else 0.0,
                    cer=cer,
                )
            )
        mode_metrics_by_strategy[strategy] = sorted(mode_metrics, key=lambda metric: metric.total, reverse=True)

        coverage_metrics: list[FailureModeCoverage] = []
        for tag, count in coverage_counts.items():
            coverage_metrics.append(
                FailureModeCoverage(
                    name=tag,
                    total=int(count),
                    rate=(count / total_samples) if total_samples else 0.0,
                )
            )
        mode_coverage_by_strategy[strategy] = sorted(coverage_metrics, key=lambda metric: metric.total, reverse=True)

        primary_metrics: list[FailureModeMetrics] = []
        for tag, stats in primary_counts.items():
            total_count = int(stats["total"])
            exact = int(stats["exact"])
            cer = stats["cer"] / total_count if total_count else 0.0
            primary_metrics.append(
                FailureModeMetrics(
                    name=tag,
                    total=total_count,
                    exact_match=exact,
                    exact_match_rate=exact / total_count if total_count else 0.0,
                    cer=cer,
                )
            )
        primary_mode_metrics_by_strategy[strategy] = sorted(
            primary_metrics, key=lambda metric: metric.total, reverse=True
        )

    return AnalysisReport(
        total=total_samples,
        dataset_words=len(dataset),
        lexicon_words=len(lexicon),
        strategies=normalized_strategies,
        metrics=metrics,
        mode_metrics_by_strategy=mode_metrics_by_strategy,
        mode_coverage_by_strategy=mode_coverage_by_strategy,
        primary_mode_metrics_by_strategy=primary_mode_metrics_by_strategy,
        samples=samples if include_samples else None,
    )


def write_analysis_csv(path: Path, samples: Sequence[AnalysisSample], strategies: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["word", "misaki", "charsiu_ipa"]
        for strategy in strategies:
            header.extend(
                [
                    f"{strategy}_mapped",
                    f"{strategy}_mapped_stress",
                    f"{strategy}_match",
                    f"{strategy}_match_stress",
                ]
            )
        writer.writerow(header)
        for sample in samples:
            row = [sample.word, sample.misaki, sample.charsiu_ipa]
            for strategy in strategies:
                mapped = sample.mapped_by_strategy.get(strategy, "")
                mapped_stress = sample.mapped_stress_by_strategy.get(strategy, "")
                row.extend(
                    [
                        mapped,
                        mapped_stress,
                        int(sample.misaki == mapped),
                        int(sample.misaki == mapped_stress),
                    ]
                )
            writer.writerow(row)


def filter_samples_by_mode(
    samples: Sequence[AnalysisSample],
    *,
    strategy: str,
    modes: Sequence[str] | None,
    primary_modes: Sequence[str] | None,
) -> list[AnalysisSample]:
    normalized_strategy = normalize_strategy(strategy)
    mode_set = {mode.strip() for mode in modes or [] if mode.strip()}
    primary_set = {mode.strip() for mode in primary_modes or [] if mode.strip()}

    filtered: list[AnalysisSample] = []
    for sample in samples:
        hyp = sample.mapped_stress_by_strategy.get(normalized_strategy)
        if hyp is None:
            raise ValueError(f"Missing mapped output for strategy: {normalized_strategy}")
        tags = classify_failure_modes(sample.misaki, hyp)
        if not tags:
            tags = {"match"}
        primary = "match" if tags == {"match"} else pick_primary_mode(tags)

        if mode_set and not (tags & mode_set):
            continue
        if primary_set and primary not in primary_set:
            continue
        filtered.append(sample)
    return filtered


def build_stress_prefix_report(
    samples: Sequence[AnalysisSample],
    *,
    strategy: str,
) -> list[StressPrefixMetrics]:
    normalized_strategy = normalize_strategy(strategy)
    stats: dict[str, dict[str, int]] = {}

    for sample in samples:
        prefix = _match_stress_prefix(sample.word)
        if prefix is None:
            continue
        hyp = sample.mapped_stress_by_strategy.get(normalized_strategy)
        if hyp is None:
            raise ValueError(f"Missing mapped output for strategy: {normalized_strategy}")
        tags = classify_failure_modes(sample.misaki, hyp)
        if not tags:
            tags = {"match"}
        primary = "match" if tags == {"match"} else pick_primary_mode(tags)
        if "stress_mismatch" not in tags and "stress_only" not in tags and primary not in {
            "stress_mismatch",
            "stress_only",
        }:
            continue

        entry = stats.setdefault(
            prefix,
            {
                "any_total": 0,
                "primary_total": 0,
                "swap_candidates": 0,
                "extra_initial_secondary": 0,
            },
        )
        if "stress_mismatch" in tags or "stress_only" in tags:
            entry["any_total"] += 1
        if primary in {"stress_mismatch", "stress_only"}:
            entry["primary_total"] += 1

        ref_sig = _stress_signature(sample.misaki)
        hyp_sig = _stress_signature(hyp)
        if (
            ref_sig.count("ˈ") == 1
            and ref_sig.count("ˌ") == 1
            and hyp_sig.count("ˈ") == 1
            and hyp_sig.count("ˌ") == 1
            and ref_sig != hyp_sig
        ):
            entry["swap_candidates"] += 1
        if _first_stress(hyp) == "ˌ" and _first_stress(sample.misaki) != "ˌ":
            entry["extra_initial_secondary"] += 1

    metrics: list[StressPrefixMetrics] = []
    for prefix, values in stats.items():
        metrics.append(
            StressPrefixMetrics(
                prefix=prefix,
                any_total=values["any_total"],
                primary_total=values["primary_total"],
                swap_candidates=values["swap_candidates"],
                extra_initial_secondary=values["extra_initial_secondary"],
            )
        )
    return sorted(metrics, key=lambda item: item.primary_total, reverse=True)
