from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from charsiug2p_tvm.config import DEFAULT_CONFIG, resolve_target


@dataclass(frozen=True)
class ExportedModules:
    encoder: "tvm.IRModule"
    decoder: "tvm.IRModule"


def default_output_dir(
    *,
    checkpoint: str,
    target: str,
    batch_size: int,
    max_input_bytes: int,
    max_output_len: int,
) -> Path:
    safe_name = checkpoint.replace("/", "_")
    details = f"b{batch_size}_in{max_input_bytes}_out{max_output_len}"
    return Path("dist") / "tvm" / safe_name / details / target


def export_torch_model(
    *,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    max_input_bytes: int = DEFAULT_CONFIG.max_input_bytes,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
) -> ExportedModules:
    """Export encoder/decoder wrappers to Relax IRModules."""
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1.")
    if max_input_bytes <= 0:
        raise ValueError("max_input_bytes must be >= 1.")
    if max_output_len <= 0:
        raise ValueError("max_output_len must be >= 1.")

    import torch
    import tvm
    from transformers import T5ForConditionalGeneration
    from tvm.relax.frontend.torch import from_exported_program

    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.eval()

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder: torch.nn.Module):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            return outputs.last_hidden_state

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder: torch.nn.Module, lm_head: torch.nn.Module, config: object, model_dim: int):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.config = config
            self.model_dim = model_dim

        def forward(
            self,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            sequence_output = outputs.last_hidden_state
            if getattr(self.config, "tie_word_embeddings", False):
                sequence_output = sequence_output * (self.model_dim**-0.5)
            return self.lm_head(sequence_output)

    encoder = EncoderWrapper(model.encoder)
    decoder = DecoderWrapper(model.decoder, model.lm_head, model.config, getattr(model, "model_dim", model.config.d_model))

    input_ids = torch.zeros((batch_size, max_input_bytes), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    encoder_ep = torch.export.export(encoder, (input_ids, attention_mask))
    encoder_mod = from_exported_program(encoder_ep, keep_params_as_input=False)

    hidden_state = torch.zeros((batch_size, max_input_bytes, model.config.d_model), dtype=torch.float32)
    decoder_input_ids = torch.zeros((batch_size, max_output_len), dtype=torch.long)
    decoder_ep = torch.export.export(decoder, (decoder_input_ids, hidden_state, attention_mask))
    decoder_mod = from_exported_program(decoder_ep, keep_params_as_input=False)

    return ExportedModules(encoder=encoder_mod, decoder=decoder_mod)


def compile_tvm_module(
    *,
    output_dir: Path | None,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    max_input_bytes: int = DEFAULT_CONFIG.max_input_bytes,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
    target: str = "llvm",
    output_ext: str | None = None,
) -> dict[str, Path]:
    """Compile encoder/decoder modules into TVM runtime artifacts."""
    resolved = resolve_target(target, output_ext=output_ext)
    target_name = resolved.name
    output_ext = resolved.output_ext
    if output_dir is None:
        output_dir = default_output_dir(
            checkpoint=checkpoint,
            target=target_name,
            batch_size=batch_size,
            max_input_bytes=max_input_bytes,
            max_output_len=max_output_len,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    import tvm
    import tvm.relax as relax

    mods = export_torch_model(
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_input_bytes=max_input_bytes,
        max_output_len=max_output_len,
    )

    artifacts: dict[str, Path] = {}
    target_obj = resolved.target
    relax_pipeline = relax.get_default_pipeline(target_obj)

    for name, mod in {"encoder": mods.encoder, "decoder": mods.decoder}.items():
        exec_obj = relax.build(
            mod,
            target=target_obj,
            exec_mode="bytecode",
            relax_pipeline=relax_pipeline,
        )
        out_path = output_dir / f"{name}.{output_ext}"
        exec_obj.export_library(str(out_path))
        artifacts[name] = out_path

        script_path = output_dir / f"{name}.relax.py"
        script_path.write_text(mod.script(show_meta=True), encoding="utf-8")

    metadata_entries = [
        f"checkpoint={checkpoint}",
        f"batch_size={batch_size}",
        f"max_input_bytes={max_input_bytes}",
        f"max_output_len={max_output_len}",
        f"target={target_name}",
        f"target_spec={target_obj}",
        f"output_ext={output_ext}",
    ]
    if resolved.export_func:
        metadata_entries.append(f"export_func={resolved.export_func}")
    metadata_path = output_dir / "compile_metadata.txt"
    metadata_path.write_text("\n".join(metadata_entries) + "\n", encoding="utf-8")

    return artifacts
