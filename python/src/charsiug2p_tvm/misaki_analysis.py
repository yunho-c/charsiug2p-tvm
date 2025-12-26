from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import csv
import json
import random

from charsiug2p_tvm.harness import reference_g2p

_FROM_ESPEAKS = sorted(
    {
        "\u0303": "",
        "a^ɪ": "I",
        "a^ʊ": "W",
        "d^ʒ": "ʤ",
        "e": "A",
        "e^ɪ": "A",
        "r": "ɹ",
        "t^ʃ": "ʧ",
        "x": "k",
        "ç": "k",
        "ɐ": "ə",
        "ɔ^ɪ": "Y",
        "ə^l": "ᵊl",
        "ɚ": "əɹ",
        "ɬ": "l",
        "ʔ": "t",
        "ʔn": "tᵊn",
        "ʔˌn\u0329": "tᵊn",
        "ʲ": "",
        "ʲO": "jO",
        "ʲQ": "jQ",
    }.items(),
    key=lambda kv: -len(kv[0]),
)

_VOWELS = {
    "ɑ",
    "ɔ",
    "ɛ",
    "ɜ",
    "ɪ",
    "ʊ",
    "ʌ",
    "ə",
    "i",
    "u",
    "A",
    "I",
    "W",
    "Y",
    "O",
    "Q",
    "æ",
    "a",
    "ɒ",
    "ᵻ",
    "ᵊ",
}


@dataclass(frozen=True)
class AnalysisSample:
    word: str
    misaki: str
    charsiu_ipa: str
    mapped: str
    mapped_stress: str


@dataclass(frozen=True)
class MappingMetrics:
    name: str
    exact_match: int
    exact_match_rate: float
    cer: float


@dataclass(frozen=True)
class AnalysisReport:
    total: int
    dataset_words: int
    lexicon_words: int
    metrics: list[MappingMetrics]
    samples: list[AnalysisSample] | None


def _swap_syllabic_markers(text: str) -> str:
    chars = list(text)
    i = 0
    while i < len(chars):
        if i + 1 < len(chars) and chars[i + 1] == "\u0329":
            consonant = chars[i]
            chars[i] = "ᵊ"
            chars[i + 1] = consonant
            i += 2
        else:
            i += 1
    return "".join(chars).replace("\u0329", "")


def espeak_ipa_to_misaki(ipa: str, *, british: bool) -> str:
    result = ipa.replace("\u0361", "^")
    for old, new in _FROM_ESPEAKS:
        result = result.replace(old, new)
    result = _swap_syllabic_markers(result)
    if british:
        result = result.replace("e^ə", "ɛː")
        result = result.replace("iə", "ɪə")
        result = result.replace("ə^ʊ", "Q")
    else:
        result = result.replace("o^ʊ", "O")
        result = result.replace("ɜːɹ", "ɜɹ")
        result = result.replace("ɜː", "ɜɹ")
        result = result.replace("ɪə", "iə")
        result = result.replace("ː", "")
    return result.replace("^", "")


def fix_stress_placement(misaki_tokens: str) -> str:
    result: list[str] = []
    pending_stress: str | None = None

    for char in misaki_tokens:
        if char in {"ˈ", "ˌ"}:
            if pending_stress:
                result.append(pending_stress)
            pending_stress = char
        elif char in _VOWELS:
            if pending_stress:
                result.append(pending_stress)
                pending_stress = None
            result.append(char)
        else:
            result.append(char)

    if pending_stress:
        result.append(pending_stress)
    return "".join(result)


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
    limit: int | None,
    shuffle: bool,
    seed: int | None,
    device: str,
    batch_size: int,
    include_samples: bool,
) -> AnalysisReport:
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

    mapped_outputs: list[str] = []
    mapped_stress_outputs: list[str] = []
    misaki_outputs: list[str] = []
    samples: list[AnalysisSample] = []

    for word, ipa in zip(words, charsiu_outputs):
        misaki = lexicon[word]
        mapped = espeak_ipa_to_misaki(ipa, british=british)
        mapped_stress = fix_stress_placement(mapped)
        mapped_outputs.append(mapped)
        mapped_stress_outputs.append(mapped_stress)
        misaki_outputs.append(misaki)
        if include_samples:
            samples.append(
                AnalysisSample(
                    word=word,
                    misaki=misaki,
                    charsiu_ipa=ipa,
                    mapped=mapped,
                    mapped_stress=mapped_stress,
                )
            )

    total = len(misaki_outputs)
    metrics: list[MappingMetrics] = []
    for name, outputs in (
        ("mapped", mapped_outputs),
        ("mapped+stress", mapped_stress_outputs),
    ):
        exact = sum(1 for ref, hyp in zip(misaki_outputs, outputs) if ref == hyp)
        cer = sum(_cer(ref, hyp) for ref, hyp in zip(misaki_outputs, outputs)) / total if total else 0.0
        metrics.append(MappingMetrics(name=name, exact_match=exact, exact_match_rate=exact / total if total else 0.0, cer=cer))

    return AnalysisReport(
        total=total,
        dataset_words=len(dataset),
        lexicon_words=len(lexicon),
        metrics=metrics,
        samples=samples if include_samples else None,
    )


def write_analysis_csv(path: Path, samples: Sequence[AnalysisSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "word",
                "misaki",
                "charsiu_ipa",
                "mapped",
                "mapped_stress",
                "match_mapped",
                "match_mapped_stress",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.word,
                    sample.misaki,
                    sample.charsiu_ipa,
                    sample.mapped,
                    sample.mapped_stress,
                    int(sample.misaki == sample.mapped),
                    int(sample.misaki == sample.mapped_stress),
                ]
            )
