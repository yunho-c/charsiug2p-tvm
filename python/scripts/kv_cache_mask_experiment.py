"""Check whether decoder_attention_mask masks padded KV-cache positions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import T5ForConditionalGeneration
from transformers.cache_utils import Cache, DynamicLayer, EncoderDecoderCache


@dataclass(frozen=True)
class ExperimentResult:
    label: str
    max_abs_diff: float


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


def _make_decoder_attention_mask(max_len: int, cur_pos: torch.Tensor, step_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=cur_pos.device)
    max_pos = cur_pos.to(torch.int64) + step_len
    valid = positions < max_pos
    return valid.to(torch.long).unsqueeze(0)


def _clone_cache_with_padding(
    base_cache: ExportEncoderDecoderCache,
    prefix_keys: list[torch.Tensor],
    prefix_values: list[torch.Tensor],
    *,
    max_cache_len: int,
    pad_random: bool,
) -> ExportEncoderDecoderCache:
    self_cache = Cache(layer_class_to_replicate=lambda: ExportDynamicLayer(max_cache_len))
    cache = ExportEncoderDecoderCache(self_cache, base_cache.cross_attention_cache, base_cache.cur_pos)
    for layer, keys, values in zip(self_cache.layers, prefix_keys, prefix_values):
        pad_len = max_cache_len - keys.shape[-2]
        if pad_len > 0:
            pad_shape = (keys.shape[0], keys.shape[1], pad_len, keys.shape[3])
            if pad_random:
                pad_k = torch.randn(pad_shape, device=keys.device, dtype=keys.dtype)
                pad_v = torch.randn(pad_shape, device=values.device, dtype=values.dtype)
            else:
                pad_k = torch.zeros(pad_shape, device=keys.device, dtype=keys.dtype)
                pad_v = torch.zeros(pad_shape, device=values.device, dtype=values.dtype)
            keys = torch.cat([keys, pad_k], dim=-2)
            values = torch.cat([values, pad_v], dim=-2)
        layer.keys = keys
        layer.values = values
        layer.dtype = keys.dtype
        layer.device = keys.device
        layer.is_initialized = True
    return cache


def _run_step(
    model: T5ForConditionalGeneration,
    decoder_input_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    past_key_values: ExportEncoderDecoderCache,
    *,
    decoder_attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    cache_position = torch.zeros((decoder_input_ids.shape[1],), device=decoder_input_ids.device, dtype=torch.int64)
    cache_position = cache_position + past_key_values.cur_pos.to(torch.int64)
    outputs = model.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    return outputs.last_hidden_state


def main() -> None:
    torch.manual_seed(0)
    checkpoint = "charsiu/g2p_multilingual_byT5_tiny_8_layers_100"
    max_output_len = 8
    max_input_len = 8

    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (1, max_input_len))
    encoder_attention_mask = torch.ones_like(input_ids)
    encoder_outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=encoder_attention_mask,
        return_dict=True,
    )
    encoder_hidden_states = encoder_outputs.last_hidden_state

    self_cache = Cache(layer_class_to_replicate=lambda: ExportDynamicLayer(max_output_len))
    cross_cache = Cache(layer_class_to_replicate=lambda: ExportDynamicLayer(max_input_len))
    cur_pos = torch.tensor(0, dtype=torch.int32)
    past_key_values = ExportEncoderDecoderCache(self_cache, cross_cache, cur_pos)

    decoder_input_ids = torch.tensor([[model.config.pad_token_id]])
    cache_position = torch.arange(decoder_input_ids.shape[1])
    past_key_values.cur_pos = cache_position[-1] + 1
    decoder_attention_mask = _make_decoder_attention_mask(max_output_len, past_key_values.cur_pos, 1)
    outputs = model.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    past = outputs.past_key_values

    prefix_keys = [layer.keys.clone() for layer in past.self_attention_cache.layers]
    prefix_values = [layer.values.clone() for layer in past.self_attention_cache.layers]
    cur_pos = torch.tensor(decoder_input_ids.shape[1], dtype=torch.int32)

    padded_clean = _clone_cache_with_padding(
        past,
        prefix_keys,
        prefix_values,
        max_cache_len=max_output_len,
        pad_random=False,
    )
    padded_noise = _clone_cache_with_padding(
        past,
        prefix_keys,
        prefix_values,
        max_cache_len=max_output_len,
        pad_random=True,
    )
    padded_clean.cur_pos = cur_pos
    padded_noise.cur_pos = cur_pos

    step_ids = torch.tensor([[model.config.pad_token_id]])
    mask = _make_decoder_attention_mask(max_output_len, cur_pos, step_ids.shape[1])

    outputs_masked_clean = _run_step(
        model,
        step_ids,
        encoder_hidden_states,
        encoder_attention_mask,
        padded_clean,
        decoder_attention_mask=mask,
    )
    outputs_masked_noise = _run_step(
        model,
        step_ids,
        encoder_hidden_states,
        encoder_attention_mask,
        padded_noise,
        decoder_attention_mask=mask,
    )
    results = [
        ExperimentResult(
            label="masked",
            max_abs_diff=(outputs_masked_clean - outputs_masked_noise).abs().max().item(),
        )
    ]
    for result in results:
        print(f"{result.label}: max_abs_diff={result.max_abs_diff:.6e}")

    try:
        outputs_unmasked_clean = _run_step(
            model,
            step_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            padded_clean,
            decoder_attention_mask=None,
        )
        outputs_unmasked_noise = _run_step(
            model,
            step_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            padded_noise,
            decoder_attention_mask=None,
        )
        diff = (outputs_unmasked_clean - outputs_unmasked_noise).abs().max().item()
        print(f"unmasked: max_abs_diff={diff:.6e}")
    except RuntimeError as exc:
        print(f"unmasked: failed with RuntimeError: {exc}")


if __name__ == "__main__":
    main()
