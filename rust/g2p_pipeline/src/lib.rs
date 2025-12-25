use std::path::Path;

use charsiug2p_g2p_core::{prepare_prefixed_words, G2pConfig, G2pError};
use charsiug2p_g2p_tokenizer::{load_tokenizer_handle, TokenizerBackend, TokenizerError, TokenizerHandle};
use charsiug2p_g2p_tvm::{tensor_from_i64, TvmArtifacts, TvmError, TvmExecutable};

mod artifacts;
pub use artifacts::{ArtifactError, ArtifactResolver, ArtifactRoots, ArtifactSpec};

#[derive(Debug)]
pub enum PipelineError {
    Tokenizer(TokenizerError),
    Core(G2pError),
    Tvm(TvmError),
    BatchSizeOverflow { expected: usize, got: usize },
    MissingEosToken,
    MissingKvArtifacts,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Tokenizer(err) => write!(f, "{err}"),
            PipelineError::Core(err) => write!(f, "{err}"),
            PipelineError::Tvm(err) => write!(f, "{err}"),
            PipelineError::BatchSizeOverflow { expected, got } => {
                write!(f, "Batch size overflow: expected {expected}, got {got}")
            }
            PipelineError::MissingEosToken => write!(f, "Missing EOS token id"),
            PipelineError::MissingKvArtifacts => write!(f, "KV-cache artifacts are missing or incomplete."),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<TokenizerError> for PipelineError {
    fn from(err: TokenizerError) -> Self {
        PipelineError::Tokenizer(err)
    }
}

impl From<G2pError> for PipelineError {
    fn from(err: G2pError) -> Self {
        PipelineError::Core(err)
    }
}

impl From<TvmError> for PipelineError {
    fn from(err: TvmError) -> Self {
        PipelineError::Tvm(err)
    }
}

#[derive(Clone)]
pub struct PipelineConfig {
    pub max_input_bytes: usize,
    pub max_output_len: usize,
    pub batch_size: usize,
    pub eos_token_id: Option<i64>,
    pub pad_token_id: Option<i64>,
    pub use_kv_cache: bool,
}

impl PipelineConfig {
    pub fn from_core(
        config: &G2pConfig,
        batch_size: usize,
        eos_token_id: Option<i64>,
        use_kv_cache: bool,
    ) -> Self {
        Self {
            max_input_bytes: config.max_input_bytes,
            max_output_len: config.max_output_len,
            batch_size,
            eos_token_id,
            pad_token_id: None,
            use_kv_cache,
        }
    }
}

pub struct G2pPipeline {
    tokenizer: TokenizerBackend,
    tvm: TvmExecutable,
    config: PipelineConfig,
}

impl G2pPipeline {
    pub fn load(
        tokenizer_metadata: impl AsRef<Path>,
        artifacts: TvmArtifacts,
        config: PipelineConfig,
    ) -> Result<Self, PipelineError> {
        let handle = load_tokenizer_handle(tokenizer_metadata, config.max_input_bytes)?;
        let TokenizerHandle { backend, metadata } = handle;
        let tokenizer = backend;
        let tvm = TvmExecutable::load(&artifacts)?;
        let config = PipelineConfig {
            eos_token_id: config.eos_token_id.or(metadata.eos_token_id),
            pad_token_id: config.pad_token_id.or(metadata.pad_token_id),
            ..config
        };
        if config.use_kv_cache && !tvm.has_kv_cache() {
            return Err(PipelineError::MissingKvArtifacts);
        }
        Ok(Self { tokenizer, tvm, config })
    }

    pub fn run(&self, words: &[String], lang: &str, space_after_colon: bool) -> Result<Vec<String>, PipelineError> {
        if words.is_empty() {
            return Ok(vec![]);
        }
        let prefixed = prepare_prefixed_words(words, lang, space_after_colon, self.config.max_input_bytes)?;
        let mut results = Vec::with_capacity(prefixed.len());
        let batch_size = self.config.batch_size;
        for chunk in prefixed.chunks(batch_size) {
            let batch_words = chunk.iter().cloned().collect::<Vec<_>>();
            if batch_words.len() > batch_size {
                return Err(PipelineError::BatchSizeOverflow {
                    expected: batch_size,
                    got: batch_words.len(),
                });
            }
            let encoded = self.tokenizer.encode_batch(&batch_words)?;
            let pad_token_id = self.config.pad_token_id.unwrap_or(0);
            let input_ids = pad_batch_i64(&encoded.input_ids, batch_size, encoded.max_length, pad_token_id)?;
            let attention_mask = pad_batch_i64(&encoded.attention_mask, batch_size, encoded.max_length, 0)?;

            let encoder_input = tensor_from_i64(&input_ids, &[batch_size as i64, encoded.max_length as i64])?;
            let encoder_mask = tensor_from_i64(&attention_mask, &[batch_size as i64, encoded.max_length as i64])?;
            let encoder_states = self.tvm.call_encoder(&encoder_input, &encoder_mask)?;

            let generated = if self.config.use_kv_cache {
                self.decode_with_cache(
                    batch_size,
                    batch_words.len(),
                    pad_token_id,
                    encoder_states,
                    encoder_mask,
                )?
            } else {
                self.decode_without_cache(
                    batch_size,
                    batch_words.len(),
                    pad_token_id,
                    encoder_states,
                    encoder_mask,
                )?
            };

            for row in 0..batch_words.len() {
                let start = row * self.config.max_output_len;
                let end = start + self.config.max_output_len;
                let decoded = self
                    .tokenizer
                    .decode_ids(&generated[start..end], true)
                    .map_err(PipelineError::Tokenizer)?;
                results.push(decoded);
            }
        }
        Ok(results)
    }

    fn decode_without_cache(
        &self,
        batch_size: usize,
        real_batch: usize,
        pad_token_id: i64,
        encoder_states: tvm_ffi::Tensor,
        encoder_mask: tvm_ffi::Tensor,
    ) -> Result<Vec<i64>, PipelineError> {
        let mut generated = vec![pad_token_id; batch_size * self.config.max_output_len];
        let eos_token_id = self.config.eos_token_id.ok_or(PipelineError::MissingEosToken)?;
        let mut finished = vec![false; batch_size];
        if real_batch < batch_size {
            for row in real_batch..batch_size {
                finished[row] = true;
            }
        }
        for step in 1..self.config.max_output_len {
            let decoder_input = tensor_from_i64(
                &generated,
                &[batch_size as i64, self.config.max_output_len as i64],
            )?;
            let logits = self.tvm.call_decoder(&decoder_input, &encoder_states, &encoder_mask)?;
            let next_tokens = argmax_last_token(
                &logits,
                batch_size,
                self.config.max_output_len,
                step - 1,
            )?;
            for row in 0..batch_size {
                let idx = row * self.config.max_output_len + step;
                let token = if finished[row] { eos_token_id } else { next_tokens[row] };
                generated[idx] = token;
                if token == eos_token_id {
                    finished[row] = true;
                }
            }
            if finished.iter().take(real_batch).all(|done| *done) {
                break;
            }
        }
        Ok(generated)
    }

    fn decode_with_cache(
        &self,
        batch_size: usize,
        real_batch: usize,
        pad_token_id: i64,
        encoder_states: tvm_ffi::Tensor,
        encoder_mask: tvm_ffi::Tensor,
    ) -> Result<Vec<i64>, PipelineError> {
        let eos_token_id = self.config.eos_token_id.ok_or(PipelineError::MissingEosToken)?;
        let mut generated = vec![pad_token_id; batch_size * self.config.max_output_len];
        let mut finished = vec![false; batch_size];
        if real_batch < batch_size {
            for row in real_batch..batch_size {
                finished[row] = true;
            }
        }

        let prefill_ids = vec![pad_token_id; batch_size];
        let prefill_tensor = tensor_from_i64(&prefill_ids, &[batch_size as i64, 1])?;
        let (logits, mut past_k, mut past_v, mut cur_pos) =
            self.tvm
                .call_decoder_prefill(&prefill_tensor, &encoder_states, &encoder_mask)?;
        let next_tokens = argmax_last_token(&logits, batch_size, 1, 0)?;
        for row in 0..batch_size {
            generated[row * self.config.max_output_len + 1] = next_tokens[row];
        }

        for step in 2..self.config.max_output_len {
            for row in 0..batch_size {
                if generated[row * self.config.max_output_len + step - 1] == eos_token_id {
                    finished[row] = true;
                }
            }
            if finished.iter().take(real_batch).all(|done| *done) {
                break;
            }
            let mut step_ids = Vec::with_capacity(batch_size);
            for row in 0..batch_size {
                step_ids.push(generated[row * self.config.max_output_len + step - 1]);
            }
            let step_tensor = tensor_from_i64(&step_ids, &[batch_size as i64, 1])?;
            let (logits, next_k, next_v, next_pos) = self.tvm.call_decoder_step(
                &step_tensor,
                &encoder_states,
                &encoder_mask,
                &past_k,
                &past_v,
                &cur_pos,
            )?;
            past_k = next_k;
            past_v = next_v;
            cur_pos = next_pos;
            let next_tokens = argmax_last_token(&logits, batch_size, 1, 0)?;
            for row in 0..batch_size {
                let idx = row * self.config.max_output_len + step;
                let token = if finished[row] { eos_token_id } else { next_tokens[row] };
                generated[idx] = token;
                if token == eos_token_id {
                    finished[row] = true;
                }
            }
        }
        Ok(generated)
    }
}

fn pad_batch_i64(
    data: &[i64],
    batch_size: usize,
    max_len: usize,
    pad_value: i64,
) -> Result<Vec<i64>, PipelineError> {
    let expected = batch_size * max_len;
    if data.len() > expected {
        return Err(PipelineError::BatchSizeOverflow {
            expected,
            got: data.len(),
        });
    }
    if data.len() == expected {
        return Ok(data.to_vec());
    }
    let mut padded = Vec::with_capacity(expected);
    padded.extend_from_slice(data);
    padded.extend(std::iter::repeat(pad_value).take(expected - data.len()));
    Ok(padded)
}

fn argmax_last_token(
    logits: &tvm_ffi::Tensor,
    batch_size: usize,
    seq_len: usize,
    step: usize,
) -> Result<Vec<i64>, PipelineError> {
    let data = logits
        .data_as_slice::<f32>()
        .map_err(TvmError::from)
        .map_err(PipelineError::Tvm)?;
    let vocab_size = data.len() / (batch_size * seq_len);
    let mut result = Vec::with_capacity(batch_size);
    for row in 0..batch_size {
        let mut best_id = 0usize;
        let mut best_value = f32::MIN;
        let base = (row * seq_len + step) * vocab_size;
        for idx in 0..vocab_size {
            let value = data[base + idx];
            if value > best_value {
                best_value = value;
                best_id = idx;
            }
        }
        result.push(best_id as i64);
    }
    Ok(result)
}
