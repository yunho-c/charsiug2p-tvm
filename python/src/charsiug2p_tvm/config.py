from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortingConfig:
    checkpoint: str
    max_input_bytes: int
    max_output_len: int
    batch_size: int
    precision: str
    decode: str


DEFAULT_CONFIG = PortingConfig(
    checkpoint="charsiu/g2p_multilingual_byT5_tiny_8_layers_100",
    max_input_bytes=64,
    max_output_len=128,
    batch_size=1,
    precision="fp32",
    decode="greedy",
)
