use std::fmt;
use std::path::{Path, PathBuf};

use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub max_length: usize,
    pub pad_id: Option<u32>,
}

impl TokenizerConfig {
    pub fn new(max_length: usize) -> Self {
        Self {
            max_length,
            pad_id: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenizedBatch {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub batch_size: usize,
    pub max_length: usize,
}

#[derive(Debug)]
pub enum TokenizerError {
    TokenizerLoad { path: PathBuf, message: String },
    Tokenize(String),
    InvalidLength { expected: usize, got: usize },
    EmptyInput,
    MetadataLoad { path: PathBuf, message: String },
    UnsupportedTokenizer(String),
    MissingTokenizerFile(PathBuf),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::TokenizerLoad { path, message } => {
                write!(f, "Failed to load tokenizer from {}: {message}", path.display())
            }
            TokenizerError::Tokenize(message) => write!(f, "Tokenizer error: {message}"),
            TokenizerError::InvalidLength { expected, got } => {
                write!(f, "Tokenizer output length {got} does not match expected {expected}")
            }
            TokenizerError::EmptyInput => write!(f, "Tokenizer input is empty"),
            TokenizerError::MetadataLoad { path, message } => {
                write!(f, "Failed to load tokenizer metadata from {}: {message}", path.display())
            }
            TokenizerError::UnsupportedTokenizer(message) => write!(f, "{message}"),
            TokenizerError::MissingTokenizerFile(path) => {
                write!(f, "Tokenizer file not found: {}", path.display())
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

pub struct G2pTokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl G2pTokenizer {
    pub fn from_file(path: impl AsRef<Path>, config: TokenizerConfig) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let mut tokenizer = Tokenizer::from_file(path).map_err(|err| TokenizerError::TokenizerLoad {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?;
        let pad_id = config.pad_id.unwrap_or(0);
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(config.max_length),
            pad_id,
            ..Default::default()
        }));
        if let Err(err) = tokenizer.with_truncation(Some(TruncationParams {
            max_length: config.max_length,
            strategy: TruncationStrategy::LongestFirst,
            ..Default::default()
        })) {
            return Err(TokenizerError::Tokenize(err.to_string()));
        }
        Ok(Self {
            tokenizer,
            max_length: config.max_length,
        })
    }

    pub fn encode_batch(&self, inputs: &[String]) -> Result<TokenizedBatch, TokenizerError> {
        if inputs.is_empty() {
            return Err(TokenizerError::EmptyInput);
        }
        let encodings = self
            .tokenizer
            .encode_batch(inputs.to_vec(), false)
            .map_err(|err| TokenizerError::Tokenize(err.to_string()))?;
        let mut input_ids = Vec::with_capacity(encodings.len() * self.max_length);
        let mut attention_mask = Vec::with_capacity(encodings.len() * self.max_length);
        for encoding in encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            if ids.len() != self.max_length {
                return Err(TokenizerError::InvalidLength {
                    expected: self.max_length,
                    got: ids.len(),
                });
            }
            if mask.len() != self.max_length {
                return Err(TokenizerError::InvalidLength {
                    expected: self.max_length,
                    got: mask.len(),
                });
            }
            input_ids.extend(ids.iter().map(|id| *id as i64));
            attention_mask.extend(mask.iter().map(|value| *value as i64));
        }
        Ok(TokenizedBatch {
            input_ids,
            attention_mask,
            batch_size: inputs.len(),
            max_length: self.max_length,
        })
    }

    pub fn decode_ids(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        let cleaned = ids
            .iter()
            .filter_map(|id| u32::try_from(*id).ok())
            .collect::<Vec<_>>();
        self.tokenizer
            .decode(&cleaned, skip_special_tokens)
            .map_err(|err| TokenizerError::Tokenize(err.to_string()))
    }
}

pub struct Byt5Tokenizer {
    max_length: usize,
    pad_id: i64,
    eos_id: i64,
    unk_id: i64,
    offset: i64,
}

impl Byt5Tokenizer {
    pub fn new(max_length: usize, pad_id: i64, eos_id: i64, unk_id: i64) -> Self {
        Self {
            max_length,
            pad_id,
            eos_id,
            unk_id,
            offset: 3,
        }
    }

    pub fn from_metadata(metadata: &TokenizerMetadata, max_length: usize) -> Self {
        let pad_id = metadata.pad_token_id.unwrap_or(0);
        let eos_id = metadata.eos_token_id.unwrap_or(1);
        let unk_id = metadata.unk_token_id.unwrap_or(2);
        let offset = metadata.byt5_offset.unwrap_or(3);
        Self {
            max_length,
            pad_id,
            eos_id,
            unk_id,
            offset,
        }
    }

    pub fn encode_batch(&self, inputs: &[String]) -> Result<TokenizedBatch, TokenizerError> {
        if inputs.is_empty() {
            return Err(TokenizerError::EmptyInput);
        }
        let mut input_ids = Vec::with_capacity(inputs.len() * self.max_length);
        let mut attention_mask = Vec::with_capacity(inputs.len() * self.max_length);
        for text in inputs {
            let bytes = text.as_bytes();
            let mut row = Vec::with_capacity(self.max_length);
            for &byte in bytes.iter().take(self.max_length) {
                row.push(self.offset + byte as i64);
            }
            let pad_len = self.max_length.saturating_sub(row.len());
            input_ids.extend(row);
            input_ids.extend(std::iter::repeat(self.pad_id).take(pad_len));
            attention_mask.extend(std::iter::repeat(1).take(self.max_length - pad_len));
            attention_mask.extend(std::iter::repeat(0).take(pad_len));
        }
        Ok(TokenizedBatch {
            input_ids,
            attention_mask,
            batch_size: inputs.len(),
            max_length: self.max_length,
        })
    }

    pub fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> String {
        let mut bytes = Vec::with_capacity(ids.len());
        for &token_id in ids {
            if skip_special_tokens && token_id == self.eos_id {
                break;
            }
            if token_id == self.pad_id || token_id == self.eos_id || token_id == self.unk_id {
                if skip_special_tokens {
                    continue;
                }
                continue;
            }
            let byte_id = token_id - self.offset;
            if (0..=255).contains(&byte_id) {
                bytes.push(byte_id as u8);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

pub enum TokenizerBackend {
    Byt5(Byt5Tokenizer),
    Json(G2pTokenizer),
}

impl TokenizerBackend {
    pub fn encode_batch(&self, inputs: &[String]) -> Result<TokenizedBatch, TokenizerError> {
        match self {
            TokenizerBackend::Byt5(tokenizer) => tokenizer.encode_batch(inputs),
            TokenizerBackend::Json(tokenizer) => tokenizer.encode_batch(inputs),
        }
    }

    pub fn decode_ids(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        match self {
            TokenizerBackend::Byt5(tokenizer) => Ok(tokenizer.decode(ids, skip_special_tokens)),
            TokenizerBackend::Json(tokenizer) => tokenizer.decode_ids(ids, skip_special_tokens),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerMetadata {
    pub tokenizer_name: String,
    pub is_fast: bool,
    pub vocab_size: Option<u32>,
    pub model_max_length: Option<u128>,
    pub pad_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub unk_token_id: Option<i64>,
    pub special_tokens: Value,
    pub files: Vec<String>,
    pub sentencepiece_model: Option<String>,
    pub byt5_offset: Option<i64>,
}

pub struct TokenizerHandle {
    pub backend: TokenizerBackend,
    pub metadata: TokenizerMetadata,
}

pub fn load_tokenizer_metadata(path: impl AsRef<Path>) -> Result<TokenizerMetadata, TokenizerError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|err| TokenizerError::MetadataLoad {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;
    serde_json::from_str(&content).map_err(|err| TokenizerError::MetadataLoad {
        path: path.to_path_buf(),
        message: err.to_string(),
    })
}

pub fn load_tokenizer_backend(
    metadata_path: impl AsRef<Path>,
    max_length: usize,
) -> Result<TokenizerBackend, TokenizerError> {
    Ok(load_tokenizer_handle(metadata_path, max_length)?.backend)
}

pub fn load_tokenizer_handle(
    metadata_path: impl AsRef<Path>,
    max_length: usize,
) -> Result<TokenizerHandle, TokenizerError> {
    let metadata_path = metadata_path.as_ref();
    let metadata = load_tokenizer_metadata(metadata_path)?;
    let base_dir = metadata_path.parent().unwrap_or_else(|| Path::new("."));
    let backend = if metadata.files.iter().any(|name| name == "tokenizer.json") {
        let tokenizer_json = base_dir.join("tokenizer.json");
        if !tokenizer_json.exists() {
            return Err(TokenizerError::MissingTokenizerFile(tokenizer_json));
        }
        let mut config = TokenizerConfig::new(max_length);
        config.pad_id = metadata.pad_token_id.map(|value| value as u32);
        let tokenizer = G2pTokenizer::from_file(tokenizer_json, config)?;
        TokenizerBackend::Json(tokenizer)
    } else if metadata.byt5_offset.is_some() || metadata.tokenizer_name.contains("byt5") {
        TokenizerBackend::Byt5(Byt5Tokenizer::from_metadata(&metadata, max_length))
    } else {
        return Err(TokenizerError::UnsupportedTokenizer(format!(
            "Tokenizer {} is not supported yet",
            metadata.tokenizer_name
        )));
    };
    Ok(TokenizerHandle { backend, metadata })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byt5_encode_single_char() {
        let tokenizer = Byt5Tokenizer::new(4, 0, 1, 2);
        let batch = tokenizer.encode_batch(&[String::from("A")]).unwrap();
        assert_eq!(batch.input_ids[0], 68);
        assert_eq!(batch.attention_mask, vec![1, 0, 0, 0]);
    }

    #[test]
    fn byt5_decode_skips_special() {
        let tokenizer = Byt5Tokenizer::new(4, 0, 1, 2);
        let output = tokenizer.decode(&[0, 68, 1, 2], true);
        assert_eq!(output, "A");
    }

    #[test]
    fn byt5_round_trip_multibyte() {
        let tokenizer = Byt5Tokenizer::new(4, 0, 1, 2);
        let batch = tokenizer.encode_batch(&[String::from("é")]).unwrap();
        assert_eq!(batch.input_ids[..4], [198, 172, 0, 0]);
        let decoded = tokenizer.decode(&batch.input_ids[..2], true);
        assert_eq!(decoded, "é");
    }
}
