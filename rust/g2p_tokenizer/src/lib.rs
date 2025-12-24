use std::fmt;
use std::path::{Path, PathBuf};

use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

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
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: config.max_length,
            strategy: TruncationStrategy::LongestFirst,
            ..Default::default()
        }));
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
}
