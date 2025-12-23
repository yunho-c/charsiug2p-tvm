from __future__ import annotations

from typing import Iterable, Sequence

from charsiug2p_tvm.config import DEFAULT_CONFIG


def add_language_prefix(words: Iterable[str], lang: str, *, space_after_colon: bool = False) -> list[str]:
    prefix = f"<{lang}>:" + (" " if space_after_colon else "")
    return [f"{prefix}{word}" for word in words]


def reference_g2p(
    words: Sequence[str],
    lang: str,
    *,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
    space_after_colon: bool = False,
) -> list[str]:
    """Reference implementation placeholder (PyTorch/transformers)."""
    raise NotImplementedError(
        "reference_g2p is not implemented yet. "
        "Add a transformers-based reference path before validating TVM outputs."
    )
