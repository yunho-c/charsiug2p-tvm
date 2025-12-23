from __future__ import annotations

from pathlib import Path

from charsiug2p_tvm.config import DEFAULT_CONFIG


def export_torch_model(*, checkpoint: str = DEFAULT_CONFIG.checkpoint) -> None:
    """Export the PyTorch model to a Relax IRModule (placeholder)."""
    raise NotImplementedError(
        "export_torch_model is not implemented yet. "
        "Use torch.export.export + tvm.relax.frontend.torch.from_exported_program."
    )


def compile_tvm_module(*, output_dir: Path, checkpoint: str = DEFAULT_CONFIG.checkpoint) -> None:
    """Compile the Relax module into a TVM runtime artifact (placeholder)."""
    raise NotImplementedError(
        "compile_tvm_module is not implemented yet. "
        "Implement target-specific compilation and artifact packaging."
    )
