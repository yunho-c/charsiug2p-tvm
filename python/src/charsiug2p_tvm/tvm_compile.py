from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from charsiug2p_tvm.config import DEFAULT_CONFIG, resolve_target


@dataclass(frozen=True)
class ExportedModules:
    encoder: "tvm.IRModule"
    decoder: "tvm.IRModule | None" = None
    decoder_prefill: "tvm.IRModule | None" = None
    decoder_step: "tvm.IRModule | None" = None


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
    """Export encoder/decoder wrappers to Relax IRModules (no KV-cache)."""
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


def export_torch_model_with_cache(
    *,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    max_input_bytes: int = DEFAULT_CONFIG.max_input_bytes,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
) -> ExportedModules:
    """Export encoder + decoder prefill/step modules with KV-cache (experimental)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1.")
    if max_input_bytes <= 0:
        raise ValueError("max_input_bytes must be >= 1.")
    if max_output_len <= 0:
        raise ValueError("max_output_len must be >= 1.")

    import torch
    from transformers import T5ForConditionalGeneration
    from transformers.cache_utils import Cache, DynamicLayer, EncoderDecoderCache
    from tvm.relax.frontend.torch import from_exported_program

    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.eval()

    num_layers = model.config.num_decoder_layers
    num_heads = model.config.num_heads
    head_dim = model.config.d_kv
    model_dim = getattr(model, "model_dim", model.config.d_model)

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder: torch.nn.Module):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            return outputs.last_hidden_state

    class ExportDynamicLayer(DynamicLayer):
        def __init__(self, max_cache_len: int):
            super().__init__()
            self.max_cache_len = max_cache_len

        def lazy_initialization(self, key_states: torch.Tensor):
            self.dtype, self.device = key_states.dtype, key_states.device
            self.keys = key_states.new_zeros(
                (key_states.shape[0], key_states.shape[1], self.max_cache_len, key_states.shape[3])
            )
            self.values = key_states.new_zeros(
                (key_states.shape[0], key_states.shape[1], self.max_cache_len, key_states.shape[3])
            )
            self.is_initialized = True

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            cache_kwargs: dict[str, object] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not self.is_initialized:
                self.lazy_initialization(key_states)

            cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
            if cache_position is None or key_states.shape[-2] != 1:
                keys, values = key_states, value_states
                if key_states.shape[-2] < self.max_cache_len:
                    pad_len = self.max_cache_len - key_states.shape[-2]
                    pad_shape = (key_states.shape[0], key_states.shape[1], pad_len, key_states.shape[3])
                    pad_k = key_states.new_zeros(pad_shape)
                    pad_v = value_states.new_zeros(pad_shape)
                    keys = torch.cat([key_states, pad_k], dim=-2)
                    values = torch.cat([value_states, pad_v], dim=-2)
                elif key_states.shape[-2] > self.max_cache_len:
                    keys = key_states[:, :, -self.max_cache_len :, :]
                    values = value_states[:, :, -self.max_cache_len :, :]
                self.keys = keys
                self.values = values
                return self.keys, self.values

            pos = cache_position.to(torch.int64).view(1, 1, 1, 1)
            positions = torch.arange(self.max_cache_len, device=key_states.device).view(1, 1, -1, 1)
            mask = positions == pos
            key_broadcast = key_states.expand(-1, -1, self.max_cache_len, -1)
            value_broadcast = value_states.expand(-1, -1, self.max_cache_len, -1)
            self.keys = torch.where(mask, key_broadcast, self.keys)
            self.values = torch.where(mask, value_broadcast, self.values)
            return self.keys, self.values

    class ExportEncoderDecoderCache(EncoderDecoderCache):
        def __init__(self, self_cache: Cache, cross_cache: Cache, cur_pos: torch.Tensor):
            self.self_attention_cache = self_cache
            self.cross_attention_cache = cross_cache
            self.cur_pos = cur_pos
            self.is_updated = {}

        def get_seq_length(self, layer_idx: int = 0) -> torch.Tensor:
            return self.cur_pos

    class DecoderPrefillWrapper(torch.nn.Module):
        def __init__(self, decoder: torch.nn.Module, lm_head: torch.nn.Module, config: object):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.config = config

        def _init_cache(self, device: torch.device, dtype: torch.dtype) -> EncoderDecoderCache:
            cur_pos = torch.tensor(0, dtype=torch.int32, device=device)
            self_cache = Cache(layer_class_to_replicate=lambda: ExportDynamicLayer(max_output_len))
            cross_cache = Cache(layer_class_to_replicate=lambda: ExportDynamicLayer(max_input_bytes))
            return ExportEncoderDecoderCache(self_cache, cross_cache, cur_pos)

        def forward(
            self,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            past_key_values = self._init_cache(encoder_hidden_states.device, encoder_hidden_states.dtype)
            positions = torch.arange(max_output_len, device=decoder_input_ids.device)
            valid = positions < decoder_input_ids.shape[1]
            decoder_attention_mask = valid.to(torch.long).unsqueeze(0).expand(decoder_input_ids.shape[0], -1)
            cache_position = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)
            past_key_values.cur_pos = cache_position[-1] + 1
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            sequence_output = outputs.last_hidden_state
            if getattr(self.config, "tie_word_embeddings", False):
                sequence_output = sequence_output * (model_dim**-0.5)
            logits = self.lm_head(sequence_output)
            past = outputs.past_key_values
            past_k = torch.stack([kv[0] for kv in past], dim=0)
            past_v = torch.stack([kv[1] for kv in past], dim=0)
            if past_k.shape[-2] < max_output_len:
                pad_len = max_output_len - past_k.shape[-2]
                pad_shape = (num_layers, past_k.shape[1], past_k.shape[2], pad_len, past_k.shape[4])
                pad_k = past_k.new_zeros(pad_shape)
                pad_v = past_v.new_zeros(pad_shape)
                past_k = torch.cat([past_k, pad_k], dim=-2)
                past_v = torch.cat([past_v, pad_v], dim=-2)
            cur_pos = torch.tensor(decoder_input_ids.shape[1], device=decoder_input_ids.device, dtype=torch.int32)
            return logits, past_k, past_v, cur_pos

    class DecoderStepWrapper(torch.nn.Module):
        def __init__(self, decoder: torch.nn.Module, lm_head: torch.nn.Module, config: object):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.config = config

        def _init_cache(self, device: torch.device, dtype: torch.dtype, cur_pos: torch.Tensor) -> EncoderDecoderCache:
            self_cache = Cache(layer_class_to_replicate=lambda: ExportDynamicLayer(max_output_len))
            cross_cache = Cache(layer_class_to_replicate=lambda: ExportDynamicLayer(max_input_bytes))
            return ExportEncoderDecoderCache(self_cache, cross_cache, cur_pos)

        def forward(
            self,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
            past_k: torch.Tensor,
            past_v: torch.Tensor,
            cur_pos: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            past_key_values = self._init_cache(encoder_hidden_states.device, encoder_hidden_states.dtype, cur_pos)
            for layer_idx in range(num_layers):
                past_key_values.self_attention_cache.update(past_k[layer_idx], past_v[layer_idx], layer_idx)
            positions = torch.arange(max_output_len, device=decoder_input_ids.device)
            max_pos = cur_pos.to(torch.int64) + decoder_input_ids.shape[1]
            valid = positions < max_pos
            decoder_attention_mask = valid.to(torch.long).unsqueeze(0).expand(decoder_input_ids.shape[0], -1)
            pos_offset = torch.zeros(
                (decoder_input_ids.shape[1],), device=decoder_input_ids.device, dtype=torch.int64
            )
            cache_position = pos_offset + cur_pos.to(torch.int64)
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            sequence_output = outputs.last_hidden_state
            if getattr(self.config, "tie_word_embeddings", False):
                sequence_output = sequence_output * (model_dim**-0.5)
            logits = self.lm_head(sequence_output)
            past = outputs.past_key_values
            new_k = torch.stack([kv[0] for kv in past], dim=0)
            new_v = torch.stack([kv[1] for kv in past], dim=0)
            next_pos = cur_pos + decoder_input_ids.shape[1]
            return logits, new_k, new_v, next_pos

    encoder = EncoderWrapper(model.encoder)
    prefill = DecoderPrefillWrapper(model.decoder, model.lm_head, model.config)
    step = DecoderStepWrapper(model.decoder, model.lm_head, model.config)

    input_ids = torch.zeros((batch_size, max_input_bytes), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    encoder_ep = torch.export.export(encoder, (input_ids, attention_mask))
    encoder_mod = from_exported_program(encoder_ep, keep_params_as_input=False)

    decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
    hidden_state = torch.zeros((batch_size, max_input_bytes, model.config.d_model), dtype=torch.float32)
    prefill_ep = torch.export.export(prefill, (decoder_input_ids, hidden_state, attention_mask))
    prefill_mod = from_exported_program(prefill_ep, keep_params_as_input=False)

    step_input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
    past_k = torch.zeros((num_layers, batch_size, num_heads, max_output_len, head_dim), dtype=torch.float32)
    past_v = torch.zeros((num_layers, batch_size, num_heads, max_output_len, head_dim), dtype=torch.float32)
    cur_pos = torch.tensor(1, dtype=torch.int32)
    step_ep = torch.export.export(step, (step_input_ids, hidden_state, attention_mask, past_k, past_v, cur_pos))
    step_mod = from_exported_program(step_ep, keep_params_as_input=False)

    return ExportedModules(
        encoder=encoder_mod,
        decoder_prefill=prefill_mod,
        decoder_step=step_mod,
    )


def compile_tvm_module(
    *,
    output_dir: Path | None,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    max_input_bytes: int = DEFAULT_CONFIG.max_input_bytes,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
    target: str = "llvm",
    output_ext: str | None = None,
    mixed_precision: bool = False,
    mixed_precision_out_dtype: str = "float32",
    fp16_input_names: list[str] | None = None,
    use_kv_cache: bool = False,
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

    if use_kv_cache:
        mods = export_torch_model_with_cache(
            checkpoint=checkpoint,
            batch_size=batch_size,
            max_input_bytes=max_input_bytes,
            max_output_len=max_output_len,
        )
    else:
        mods = export_torch_model(
            checkpoint=checkpoint,
            batch_size=batch_size,
            max_input_bytes=max_input_bytes,
            max_output_len=max_output_len,
        )

    artifacts: dict[str, Path] = {}
    target_obj = resolved.target
    relax_pipeline = relax.get_default_pipeline(target_obj)
    if mixed_precision:
        fp16_input_names = fp16_input_names or None
        mixed_precision_pipeline = tvm.transform.Sequential(
            [
                relax.transform.ConvertToDataflow(),
                relax.transform.ToMixedPrecision(
                    out_dtype=mixed_precision_out_dtype,
                    fp16_input_names=fp16_input_names,
                ),
                relax.transform.ToNonDataflow(),
            ]
        )
        relax_pipeline = tvm.transform.Sequential([mixed_precision_pipeline, relax_pipeline])

    module_map = {"encoder": mods.encoder}
    if mods.decoder is not None:
        module_map["decoder"] = mods.decoder
    if mods.decoder_prefill is not None:
        module_map["decoder_prefill"] = mods.decoder_prefill
    if mods.decoder_step is not None:
        module_map["decoder_step"] = mods.decoder_step

    for name, mod in module_map.items():
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
        f"use_kv_cache={use_kv_cache}",
        f"mixed_precision={mixed_precision}",
        f"mixed_precision_out_dtype={mixed_precision_out_dtype}",
        f"mixed_precision_fp16_inputs={','.join(fp16_input_names or [])}",
    ]
    if resolved.export_func:
        metadata_entries.append(f"export_func={resolved.export_func}")
    metadata_path = output_dir / "compile_metadata.txt"
    metadata_path.write_text("\n".join(metadata_entries) + "\n", encoding="utf-8")

    return artifacts
