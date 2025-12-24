from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from charsiug2p_tvm.config import default_device_for_target
from charsiug2p_tvm.eval import prepare_samples
from charsiug2p_tvm.tvm_runtime import tvm_g2p, tvm_g2p_timed


@dataclass(frozen=True)
class ProfileResult:
    target: str
    device: str
    samples: int
    total_seconds: float
    per_sample_ms: float
    encoder_seconds: float
    decoder_seconds: float
    encoder_per_sample_ms: float
    decoder_per_sample_ms: float


def _default_device_for_target(target: str) -> str:
    return default_device_for_target(target)


def profile_targets(
    *,
    data_path: Path,
    targets: Sequence[str],
    limit: int | None,
    shuffle: bool,
    seed: int | None,
    checkpoint: str,
    max_input_bytes: int,
    max_output_len: int,
    space_after_colon: bool,
    tvm_output_ext: str | None,
    tvm_batch_size: int,
    runs: int,
    warmup: bool,
    device: str | None,
) -> list[ProfileResult]:
    samples = prepare_samples(
        path=data_path,
        language=None,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
    )
    if not samples:
        return []

    words = [sample.word for sample in samples]
    lang = samples[0].lang

    results: list[ProfileResult] = []
    for target in targets:
        device_name = device or _default_device_for_target(target)
        if warmup:
            tvm_g2p(
                words,
                lang,
                checkpoint=checkpoint,
                target=target,
                output_ext=tvm_output_ext,
                batch_size=tvm_batch_size,
                max_input_bytes=max_input_bytes,
                max_output_len=max_output_len,
                space_after_colon=space_after_colon,
                device=device_name,
            )

        total = 0.0
        encoder_total = 0.0
        decoder_total = 0.0
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            _, timing = tvm_g2p_timed(
                words,
                lang,
                checkpoint=checkpoint,
                target=target,
                output_ext=tvm_output_ext,
                batch_size=tvm_batch_size,
                max_input_bytes=max_input_bytes,
                max_output_len=max_output_len,
                space_after_colon=space_after_colon,
                device=device_name,
            )
            total += time.perf_counter() - start
            encoder_total += timing.encoder_seconds
            decoder_total += timing.decoder_seconds

        total_avg = total / max(1, runs)
        per_sample_ms = (total_avg / len(words)) * 1000.0
        encoder_avg = encoder_total / max(1, runs)
        decoder_avg = decoder_total / max(1, runs)
        encoder_per_sample_ms = (encoder_avg / len(words)) * 1000.0
        decoder_per_sample_ms = (decoder_avg / len(words)) * 1000.0
        results.append(
            ProfileResult(
                target=target,
                device=device_name,
                samples=len(words),
                total_seconds=total_avg,
                per_sample_ms=per_sample_ms,
                encoder_seconds=encoder_avg,
                decoder_seconds=decoder_avg,
                encoder_per_sample_ms=encoder_per_sample_ms,
                decoder_per_sample_ms=decoder_per_sample_ms,
            )
        )

    return results


def parse_targets(values: Sequence[str]) -> list[str]:
    targets: list[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        targets.extend(parts)
    if not targets:
        raise ValueError("No targets provided.")
    return targets


def write_profile_csv(path: Path, results: Sequence[ProfileResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "target",
                "device",
                "samples",
                "total_seconds",
                "per_sample_ms",
                "encoder_seconds",
                "decoder_seconds",
                "encoder_per_sample_ms",
                "decoder_per_sample_ms",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.target,
                    result.device,
                    result.samples,
                    f"{result.total_seconds:.6f}",
                    f"{result.per_sample_ms:.3f}",
                    f"{result.encoder_seconds:.6f}",
                    f"{result.decoder_seconds:.6f}",
                    f"{result.encoder_per_sample_ms:.3f}",
                    f"{result.decoder_per_sample_ms:.3f}",
                ]
            )
