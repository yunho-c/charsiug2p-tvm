from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class TargetPreset(NamedTuple):
    target: str
    extension: str
    description: str
    target_host: str | None = None
    export_func: str | None = None


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

PRETRAINED_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("charsiu/g2p_multilingual_byT5_tiny_8_layers_100", "ByT5 tiny (8-layer), 100 languages"),
    ("charsiu/g2p_multilingual_byT5_tiny_12_layers_100", "ByT5 tiny (12-layer), 100 languages"),
    ("charsiu/g2p_multilingual_byT5_tiny_16_layers_100", "ByT5 tiny (16-layer), 100 languages"),
    ("charsiu/g2p_multilingual_byT5_small_100", "ByT5 small, 100 languages"),
)

TARGET_CONFIGS: dict[str, TargetPreset] = {
    "llvm": TargetPreset(
        target="llvm -opt-level=3",
        extension="so",
        description="CPU (LLVM)",
    ),
    "metal-macos": TargetPreset(
        target_host="llvm -mtriple=arm64-apple-macos",
        target="metal",
        extension="dylib",
        description="macOS (Metal GPU + ARM64 CPU)",
    ),
    "metal-ios": TargetPreset(
        # TODO: arm64-apple-ios triple is producing macOS Mach-O for iOS builds.
        # "target_host": "llvm -mtriple=arm64-apple-ios",
        target_host="llvm -mtriple=aarch64-apple-ios",
        target="metal",
        extension="dylib",
        description="iOS (Metal GPU + ARM64 CPU)",
    ),
    "webgpu": TargetPreset(
        target_host="llvm -mtriple=wasm32-unknown-unknown-wasm",
        target="webgpu",
        extension="wasm",
        description="WebGPU (Browser via WASM)",
        export_func="tvmjs",
    ),
}


class ResolvedTarget(NamedTuple):
    name: str
    target: "tvm.target.Target"
    output_ext: str
    description: str
    export_func: str | None
    is_preset: bool


def resolve_target(target_name: str, *, output_ext: str | None = None) -> ResolvedTarget:
    """Resolve a target preset or raw target string into TVM target metadata."""
    import tvm

    preset = TARGET_CONFIGS.get(target_name)
    if preset is not None:
        host = tvm.target.Target(preset.target_host) if preset.target_host else None
        target = tvm.target.Target(preset.target, host=host)
        resolved_ext = _normalize_extension(output_ext) or preset.extension
        return ResolvedTarget(
            name=target_name,
            target=target,
            output_ext=resolved_ext,
            description=preset.description,
            export_func=preset.export_func,
            is_preset=True,
        )

    target = tvm.target.Target(target_name)
    resolved_ext = _normalize_extension(output_ext) or _default_extension_for_target(target)
    return ResolvedTarget(
        name=target_name,
        target=target,
        output_ext=resolved_ext,
        description=f"Custom target ({target_name})",
        export_func=None,
        is_preset=False,
    )


def _default_extension_for_target(target: "tvm.target.Target") -> str:
    if target.kind.name == "metal":
        return "dylib"
    if target.kind.name == "webgpu":
        return "wasm"
    return "so"


def _normalize_extension(output_ext: str | None) -> str | None:
    if output_ext is None:
        return None
    return output_ext.lstrip(".")


def default_device_for_target(target_name: str) -> str:
    target = resolve_target(target_name).target
    kind = target.kind.name
    if kind in {"cuda", "rocm", "metal", "vulkan", "opencl", "webgpu"}:
        return kind
    return "cpu"
