from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import csv
import json
import random

from charsiug2p_tvm.harness import reference_g2p
from charsiug2p_tvm.post_processing import (
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
class AnalysisReport:
    total: int
    dataset_words: int
    lexicon_words: int
    strategies: list[str]
    metrics: list[MappingMetrics]
    mode_metrics_by_strategy: dict[str, list[FailureModeMetrics]]
    samples: list[AnalysisSample] | None


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


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    return any(token in text for token in tokens)


def classify_failure_modes(ref: str, hyp: str) -> set[str]:
    if ref == hyp:
        return set()

    tags: set[str] = set()

    ref_no_stress = _strip_stress(ref)
    hyp_no_stress = _strip_stress(hyp)
    if ref_no_stress == hyp_no_stress:
        tags.add("stress_only")
    elif _contains_any(ref, ("ˈ", "ˌ")) or _contains_any(hyp, ("ˈ", "ˌ")):
        tags.add("stress_mismatch")

    if _contains_any(ref, ("ᵊ",)) or _contains_any(hyp, ("ᵊ",)):
        tags.add("syllabic_schwa")

    if _contains_any(ref, ("ɝ", "ɚ", "ɜɹ", "əɹ")) or _contains_any(hyp, ("ɝ", "ɚ", "ɜɹ", "əɹ")):
        tags.add("rhotic_vowel")

    if _contains_any(ref, ("A", "I", "W", "O", "Y", "Q")) or _contains_any(hyp, ("A", "I", "W", "O", "Y", "Q")):
        tags.add("diphthong_token")

    if _contains_any(ref, ("ʧ", "ʤ")) or _contains_any(hyp, ("ʧ", "ʤ")):
        tags.add("affricate_token")

    if _contains_any(ref, ("ɾ",)) or _contains_any(hyp, ("ɾ",)):
        tags.add("flap")

    if _contains_any(ref, ("ə", "ɜ", "ɪ", "ʌ", "ᵻ", "i")) or _contains_any(hyp, ("ə", "ɜ", "ɪ", "ʌ", "ᵻ", "i")):
        tags.add("reduced_vowel")

    if not tags:
        tags.add("other")
    return tags


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
            mapped = espeak_ipa_to_misaki(ipa, british=british, strategy=strategy)
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

    total = len(misaki_outputs)
    metrics: list[MappingMetrics] = []
    for strategy in normalized_strategies:
        for suffix, outputs in (
            ("", mapped_outputs[strategy]),
            ("+stress", mapped_stress_outputs[strategy]),
        ):
            name = f"{strategy}{suffix}"
            exact = sum(1 for ref, hyp in zip(misaki_outputs, outputs) if ref == hyp)
            cer = sum(_cer(ref, hyp) for ref, hyp in zip(misaki_outputs, outputs)) / total if total else 0.0
            metrics.append(
                MappingMetrics(
                    name=name,
                    exact_match=exact,
                    exact_match_rate=exact / total if total else 0.0,
                    cer=cer,
                )
            )

    mode_metrics_by_strategy: dict[str, list[FailureModeMetrics]] = {}
    for strategy in normalized_strategies:
        mode_counts: dict[str, dict[str, float]] = {}
        for ref, hyp in zip(misaki_outputs, mapped_stress_outputs[strategy]):
            tags = classify_failure_modes(ref, hyp)
            if not tags:
                continue
            cer = _cer(ref, hyp)
            for tag in tags:
                stats = mode_counts.setdefault(tag, {"total": 0.0, "exact": 0.0, "cer": 0.0})
                stats["total"] += 1.0
                stats["exact"] += 1.0 if ref == hyp else 0.0
                stats["cer"] += cer
        mode_metrics: list[FailureModeMetrics] = []
        for tag, stats in mode_counts.items():
            total = int(stats["total"])
            exact = int(stats["exact"])
            cer = stats["cer"] / total if total else 0.0
            mode_metrics.append(
                FailureModeMetrics(
                    name=tag,
                    total=total,
                    exact_match=exact,
                    exact_match_rate=exact / total if total else 0.0,
                    cer=cer,
                )
            )
        mode_metrics_by_strategy[strategy] = sorted(mode_metrics, key=lambda metric: metric.total, reverse=True)

    return AnalysisReport(
        total=total,
        dataset_words=len(dataset),
        lexicon_words=len(lexicon),
        strategies=normalized_strategies,
        metrics=metrics,
        mode_metrics_by_strategy=mode_metrics_by_strategy,
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
