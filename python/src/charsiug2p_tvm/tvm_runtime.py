from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

import tvm
import tvm.relax as relax
from transformers import AutoTokenizer

from charsiug2p_tvm.config import DEFAULT_CONFIG, resolve_target
from charsiug2p_tvm.harness import add_language_prefix
from charsiug2p_tvm.tvm_compile import default_output_dir

_DEFAULT_TOKENIZER = "google/byt5-small"


@dataclass(frozen=True)
class RuntimeArtifacts:
    encoder: Path
    decoder: Path


@dataclass(frozen=True)
class TvmTiming:
    encoder_seconds: float
    decoder_seconds: float


def resolve_artifacts(
    *,
    output_dir: Path | None,
    checkpoint: str,
    target: str,
    batch_size: int,
    max_input_bytes: int,
    max_output_len: int,
    output_ext: str,
) -> RuntimeArtifacts:
    if output_dir is None:
        output_dir = default_output_dir(
            checkpoint=checkpoint,
            target=target,
            batch_size=batch_size,
            max_input_bytes=max_input_bytes,
            max_output_len=max_output_len,
        )
    encoder_path = output_dir / f"encoder.{output_ext}"
    decoder_path = output_dir / f"decoder.{output_ext}"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder artifact not found: {encoder_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder artifact not found: {decoder_path}")
    return RuntimeArtifacts(encoder=encoder_path, decoder=decoder_path)


def _to_tensor(array: np.ndarray, device: tvm.runtime.Device) -> tvm.runtime.Tensor:
    tensor = tvm.runtime.empty(array.shape, dtype=str(array.dtype), device=device)
    tensor.copyfrom(array)
    return tensor


def _unwrap_single(output: object) -> tvm.runtime.Tensor:
    if isinstance(output, tvm.runtime.Tensor):
        return output
    if hasattr(output, "__len__"):
        length = len(output)  # type: ignore[arg-type]
        if length != 1:
            raise ValueError(f"Expected a single output tensor, got {length}.")
        return output[0]  # type: ignore[index]
    raise TypeError(f"Unexpected output type: {type(output)}")


class TvmG2P:
    def __init__(self, encoder_path: Path, decoder_path: Path, *, device: str = "cpu") -> None:
        self.device = tvm.runtime.device(device, 0)
        self.encoder_vm = relax.VirtualMachine(tvm.runtime.load_module(str(encoder_path)), self.device)
        self.decoder_vm = relax.VirtualMachine(tvm.runtime.load_module(str(decoder_path)), self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_TOKENIZER)

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> tvm.runtime.Tensor:
        input_tensor = _to_tensor(input_ids, self.device)
        mask_tensor = _to_tensor(attention_mask, self.device)
        output = self.encoder_vm["main"](input_tensor, mask_tensor)
        return _unwrap_single(output)

    def decode_logits(
        self,
        decoder_input_ids: np.ndarray,
        encoder_hidden_states: tvm.runtime.Tensor,
        encoder_attention_mask: np.ndarray,
    ) -> tvm.runtime.Tensor:
        decoder_tensor = _to_tensor(decoder_input_ids, self.device)
        mask_tensor = _to_tensor(encoder_attention_mask, self.device)
        output = self.decoder_vm["main"](decoder_tensor, encoder_hidden_states, mask_tensor)
        return _unwrap_single(output)


def _tvm_g2p_impl(
    words: Sequence[str],
    lang: str,
    *,
    output_dir: Path | None = None,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    target: str = "llvm",
    output_ext: str | None = None,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    max_input_bytes: int = DEFAULT_CONFIG.max_input_bytes,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
    space_after_colon: bool = False,
    device: str = "cpu",
    collect_timing: bool = False,
) -> tuple[list[str], TvmTiming | None]:
    if not words:
        return [], TvmTiming(0.0, 0.0) if collect_timing else None

    prefixed_words = add_language_prefix(words, lang, space_after_colon=space_after_colon)
    oversized = [
        (word, len(prefixed.encode("utf-8")))
        for word, prefixed in zip(words, prefixed_words)
        if len(prefixed.encode("utf-8")) > max_input_bytes
    ]
    if oversized:
        formatted = ", ".join(f"{word}={size}" for word, size in oversized[:5])
        raise ValueError(
            f"Input exceeds max_input_bytes={max_input_bytes} (examples: {formatted}). "
            "Increase the bound or filter long inputs."
        )

    resolved = resolve_target(target, output_ext=output_ext)
    target_name = resolved.name
    output_ext = resolved.output_ext

    artifacts = resolve_artifacts(
        output_dir=output_dir,
        checkpoint=checkpoint,
        target=target_name,
        batch_size=batch_size,
        max_input_bytes=max_input_bytes,
        max_output_len=max_output_len,
        output_ext=output_ext,
    )
    runtime = TvmG2P(artifacts.encoder, artifacts.decoder, device=device)

    tokenizer = runtime.tokenizer
    decoder_start = tokenizer.pad_token_id
    if decoder_start is None:
        raise ValueError("Tokenizer pad_token_id is required for decoder start.")
    eos_token = tokenizer.eos_token_id

    def pad_to_batch(array: np.ndarray, pad_value: int) -> np.ndarray:
        """Pad a smaller batch up to the compiled batch_size."""
        if array.shape[0] == batch_size:
            return array
        if array.shape[0] > batch_size:
            raise ValueError(f"Batch size overflow: {array.shape[0]} > {batch_size}.")
        pad_rows = batch_size - array.shape[0]
        pad_shape = (pad_rows,) + array.shape[1:]
        pad = np.full(pad_shape, pad_value, dtype=array.dtype)
        return np.concatenate([array, pad], axis=0)

    encoder_seconds = 0.0
    decoder_seconds = 0.0
    results: list[str] = []
    # Run in fixed-size chunks because the compiled model has a static batch size.
    for start in range(0, len(prefixed_words), batch_size):
        batch_words = prefixed_words[start : start + batch_size]
        encoded = tokenizer(
            batch_words,
            padding="max_length",
            truncation=True,
            max_length=max_input_bytes,
            add_special_tokens=False,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype("int64")
        attention_mask = encoded["attention_mask"].astype("int64")

        real_batch = input_ids.shape[0]
        # Pad to the compiled batch size so shapes match the compiled artifacts.
        input_ids = pad_to_batch(input_ids, pad_value=0)
        attention_mask = pad_to_batch(attention_mask, pad_value=0)

        if collect_timing:
            start_time = perf_counter()
            encoder_hidden_states = runtime.encode(input_ids, attention_mask)
            encoder_seconds += perf_counter() - start_time
        else:
            encoder_hidden_states = runtime.encode(input_ids, attention_mask)

        # Greedy decode: fill a fixed-length buffer with next-token argmax.
        decoder_input_ids = np.full((batch_size, max_output_len), decoder_start, dtype="int64")
        finished = np.zeros((batch_size,), dtype=bool)
        for step in range(1, max_output_len):
            if collect_timing:
                start_time = perf_counter()
                logits = runtime.decode_logits(decoder_input_ids, encoder_hidden_states, attention_mask)
                logits_np = logits.numpy()
                decoder_seconds += perf_counter() - start_time
            else:
                logits = runtime.decode_logits(decoder_input_ids, encoder_hidden_states, attention_mask)
                logits_np = logits.numpy()
            next_token = logits_np[:, step - 1].argmax(axis=-1)
            if eos_token is not None:
                next_token = np.where(finished, eos_token, next_token)
            decoder_input_ids[:, step] = next_token
            if eos_token is not None:
                finished |= next_token == eos_token
                # Stop early once all *real* rows are finished.
                if bool(finished[:real_batch].all()):
                    break

        decoded = tokenizer.batch_decode(decoder_input_ids[:, : max_output_len], skip_special_tokens=True)
        # Drop padded rows and keep only real outputs.
        results.extend(decoded[:real_batch])

    timing = TvmTiming(encoder_seconds, decoder_seconds) if collect_timing else None
    return results, timing


def tvm_g2p(
    words: Sequence[str],
    lang: str,
    *,
    output_dir: Path | None = None,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    target: str = "llvm",
    output_ext: str | None = None,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    max_input_bytes: int = DEFAULT_CONFIG.max_input_bytes,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
    space_after_colon: bool = False,
    device: str = "cpu",
) -> list[str]:
    results, _timing = _tvm_g2p_impl(
        words,
        lang,
        output_dir=output_dir,
        checkpoint=checkpoint,
        target=target,
        output_ext=output_ext,
        batch_size=batch_size,
        max_input_bytes=max_input_bytes,
        max_output_len=max_output_len,
        space_after_colon=space_after_colon,
        device=device,
        collect_timing=False,
    )
    return results


def tvm_g2p_timed(
    words: Sequence[str],
    lang: str,
    *,
    output_dir: Path | None = None,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    target: str = "llvm",
    output_ext: str | None = None,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    max_input_bytes: int = DEFAULT_CONFIG.max_input_bytes,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
    space_after_colon: bool = False,
    device: str = "cpu",
) -> tuple[list[str], TvmTiming]:
    results, timing = _tvm_g2p_impl(
        words,
        lang,
        output_dir=output_dir,
        checkpoint=checkpoint,
        target=target,
        output_ext=output_ext,
        batch_size=batch_size,
        max_input_bytes=max_input_bytes,
        max_output_len=max_output_len,
        space_after_colon=space_after_colon,
        device=device,
        collect_timing=True,
    )
    if timing is None:
        timing = TvmTiming(0.0, 0.0)
    return results, timing
