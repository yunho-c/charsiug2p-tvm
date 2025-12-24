from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from transformers import AutoTokenizer

from charsiug2p_tvm.config import DEFAULT_CONFIG

_DEFAULT_TOKENIZER = "google/byt5-small"


@dataclass(frozen=True)
class TokenizerMetadata:
    tokenizer_name: str
    is_fast: bool
    vocab_size: int | None
    model_max_length: int | None
    pad_token_id: int | None
    eos_token_id: int | None
    special_tokens: dict[str, str | list[str]]
    files: list[str]
    sentencepiece_model: str | None
    byt5_offset: int | None


@dataclass(frozen=True)
class TokenizerExportResult:
    output_dir: Path
    metadata_path: Path
    tokenizer_json: Path | None
    tokenizer_name: str


def export_tokenizer_assets(
    output_dir: Path,
    *,
    tokenizer_name: str = _DEFAULT_TOKENIZER,
    use_fast: bool = True,
    ensure_tokenizer_json: bool = True,
    export_sentencepiece: bool = True,
) -> TokenizerExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast)
    saved_files = [Path(path) for path in tokenizer.save_pretrained(output_dir)]

    tokenizer_json = output_dir / "tokenizer.json"
    if ensure_tokenizer_json and not tokenizer_json.exists():
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            backend.save(str(tokenizer_json))
            saved_files.append(tokenizer_json)
        else:
            tokenizer_json = None

    if export_sentencepiece:
        slow_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        slow_files = [Path(path) for path in slow_tokenizer.save_pretrained(output_dir)]
        saved_files.extend(slow_files)

    sentencepiece_model = _find_sentencepiece_model(output_dir)
    byt5_offset = getattr(tokenizer, "offset", None)
    if not isinstance(byt5_offset, int):
        byt5_offset = None
    metadata = TokenizerMetadata(
        tokenizer_name=tokenizer_name,
        is_fast=bool(getattr(tokenizer, "is_fast", False)),
        vocab_size=int(tokenizer.vocab_size) if getattr(tokenizer, "vocab_size", None) is not None else None,
        model_max_length=int(tokenizer.model_max_length)
        if getattr(tokenizer, "model_max_length", None) is not None
        else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        special_tokens=tokenizer.special_tokens_map,
        files=sorted({path.name for path in saved_files}),
        sentencepiece_model=sentencepiece_model.name if sentencepiece_model else None,
        byt5_offset=byt5_offset,
    )
    metadata_path = output_dir / "tokenizer_metadata.json"
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")

    return TokenizerExportResult(
        output_dir=output_dir,
        metadata_path=metadata_path,
        tokenizer_json=tokenizer_json if tokenizer_json and tokenizer_json.exists() else None,
        tokenizer_name=tokenizer_name,
    )


def default_tokenizer_export_dir(
    *,
    checkpoint: str = DEFAULT_CONFIG.checkpoint,
    max_input_bytes: int = DEFAULT_CONFIG.max_input_bytes,
    max_output_len: int = DEFAULT_CONFIG.max_output_len,
) -> Path:
    safe_name = checkpoint.replace("/", "_")
    details = f"in{max_input_bytes}_out{max_output_len}"
    return Path("dist") / "tokenizers" / safe_name / details


def _find_sentencepiece_model(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("*.model"))
    if not candidates:
        return None
    return candidates[0]
