from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

from charsiug2p_tvm.config import DEFAULT_CONFIG

_DEFAULT_TOKENIZER = "google/byt5-small"


def add_language_prefix(words: Iterable[str], lang: str, *, space_after_colon: bool = False) -> list[str]:
    prefix = f"<{lang}>:" + (" " if space_after_colon else "")
    return [f"{prefix}{word}" for word in words]


@lru_cache(maxsize=2)
def _load_reference_assets(checkpoint: str) -> tuple["AutoTokenizer", "T5ForConditionalGeneration"]:
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_TOKENIZER)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.eval()
    return tokenizer, model


def reference_g2p(
    words: Sequence[str],
    lang: str,
    *,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    max_input_bytes: int = DEFAULT_CONFIG.max_input_bytes,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
    space_after_colon: bool = False,
    device: str = "cpu",
) -> list[str]:
    """Reference implementation using transformers for validation."""
    if not words:
        return []

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

    tokenizer, model = _load_reference_assets(checkpoint)

    import torch

    inputs = tokenizer(prefixed_words, padding=True, add_special_tokens=False, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model = model.to(device)

    with torch.no_grad():
        preds = model.generate(**inputs, num_beams=1, max_length=max_output_len)
    return tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
