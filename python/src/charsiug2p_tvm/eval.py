from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import random

from charsiug2p_tvm.harness import reference_g2p
from charsiug2p_tvm.tvm_runtime import tvm_g2p


@dataclass(frozen=True)
class EvalSample:
    lang: str
    word: str
    pron: str


@dataclass(frozen=True)
class EvalMetrics:
    total: int
    exact_match: int
    exact_match_rate: float
    cer: float


def _iter_tsv_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix == ".tsv")
    return [path]


def load_eval_samples(path: Path, *, language: str | None = None) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for tsv in _iter_tsv_files(path):
        lang = language or tsv.stem
        with tsv.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                word, pron = parts[0], parts[1]
                if "," in pron:
                    pron = pron.split(",", 1)[0]
                samples.append(EvalSample(lang=lang, word=word, pron=pron))
    return samples


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


def _batch_words(words: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(words), batch_size):
        yield list(words[start : start + batch_size])


def evaluate_against_reference(
    samples: Sequence[EvalSample],
    *,
    checkpoint: str,
    max_input_bytes: int,
    max_output_len: int,
    space_after_colon: bool,
    strip_spaces: bool,
    tvm_output_dir: Path | None,
    tvm_target: str,
    tvm_output_ext: str | None,
    tvm_batch_size: int,
    ref_batch_size: int,
    ref_device: str,
    tvm_device: str,
) -> EvalMetrics:
    if not samples:
        return EvalMetrics(total=0, exact_match=0, exact_match_rate=0.0, cer=0.0)

    by_lang: dict[str, list[EvalSample]] = defaultdict(list)
    for sample in samples:
        by_lang[sample.lang].append(sample)

    ref_outputs: dict[tuple[str, str], str] = {}
    tvm_outputs: dict[tuple[str, str], str] = {}

    for lang, lang_samples in by_lang.items():
        words = [s.word for s in lang_samples]

        ref_results: list[str] = []
        for batch in _batch_words(words, ref_batch_size):
            ref_results.extend(
                reference_g2p(
                    batch,
                    lang,
                    checkpoint=checkpoint,
                    max_input_bytes=max_input_bytes,
                    max_output_len=max_output_len,
                    space_after_colon=space_after_colon,
                    device=ref_device,
                )
            )

        tvm_results: list[str] = []
        for batch in _batch_words(words, max(ref_batch_size, tvm_batch_size)):
            tvm_results.extend(
                tvm_g2p(
                    batch,
                    lang,
                    output_dir=tvm_output_dir,
                    checkpoint=checkpoint,
                    target=tvm_target,
                    output_ext=tvm_output_ext,
                    batch_size=tvm_batch_size,
                    max_input_bytes=max_input_bytes,
                    max_output_len=max_output_len,
                    space_after_colon=space_after_colon,
                    device=tvm_device,
                )
            )

        for sample, ref, tvm in zip(lang_samples, ref_results, tvm_results):
            ref_outputs[(sample.lang, sample.word)] = ref
            tvm_outputs[(sample.lang, sample.word)] = tvm

    total = len(samples)
    exact = 0
    cer_sum = 0.0
    for sample in samples:
        key = (sample.lang, sample.word)
        ref = ref_outputs[key]
        hyp = tvm_outputs[key]
        if strip_spaces:
            ref = ref.replace(" ", "")
            hyp = hyp.replace(" ", "")
        if ref == hyp:
            exact += 1
        cer_sum += _cer(ref, hyp)

    return EvalMetrics(
        total=total,
        exact_match=exact,
        exact_match_rate=exact / total if total else 0.0,
        cer=cer_sum / total if total else 0.0,
    )


def prepare_samples(
    *,
    path: Path,
    language: str | None,
    limit: int | None,
    shuffle: bool,
    seed: int | None,
) -> list[EvalSample]:
    samples = load_eval_samples(path, language=language)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    if limit is not None:
        samples = samples[:limit]
    return samples
