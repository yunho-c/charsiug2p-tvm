use std::path::PathBuf;
use std::sync::Mutex;

use flutter_rust_bridge::frb;
use serde::{Deserialize, Serialize};

use charsiug2p_g2p_core::{G2pConfig, DEFAULT_MAX_INPUT_BYTES, DEFAULT_MAX_OUTPUT_LEN};
use charsiug2p_g2p_pipeline::{
    ArtifactError, ArtifactResolver, ArtifactRoots, ArtifactSpec, G2pPipeline, PipelineConfig,
    PipelineError,
};
use charsiug2p_g2p_tokenizer::{load_tokenizer_metadata, TokenizerError, TokenizerMetadata};
use charsiug2p_g2p_tvm::{DeviceConfig, TvmError};

const DEFAULT_CHECKPOINT: &str = "charsiu/g2p_multilingual_byT5_tiny_8_layers_100";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pModelConfig {
    pub asset_root: String,
    pub checkpoint: String,
    pub target: String,
    pub max_input_bytes: u32,
    pub max_output_len: u32,
    pub batch_size: u32,
    pub tvm_ext: Option<String>,
    pub use_kv_cache: bool,
    pub device: Option<String>,
    pub device_id: i32,
    pub tokenizer_root: Option<String>,
    pub tvm_root: Option<String>,
}

impl Default for G2pModelConfig {
    fn default() -> Self {
        Self {
            asset_root: String::new(),
            checkpoint: DEFAULT_CHECKPOINT.to_string(),
            target: default_target_for_platform().to_string(),
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES as u32,
            max_output_len: DEFAULT_MAX_OUTPUT_LEN as u32,
            batch_size: 1,
            tvm_ext: None,
            use_kv_cache: true,
            device: None,
            device_id: 0,
            tokenizer_root: None,
            tvm_root: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pRunOptions {
    pub space_after_colon: bool,
}

impl Default for G2pRunOptions {
    fn default() -> Self {
        Self {
            space_after_colon: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pPlatformDefaults {
    pub target: String,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum G2pErrorKind {
    Config,
    Artifact,
    Tokenizer,
    Tvm,
    Device,
    Inference,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pFfiError {
    pub kind: G2pErrorKind,
    pub message: String,
    pub details: Option<String>,
}

impl G2pFfiError {
    fn new(kind: G2pErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            details: None,
        }
    }

    fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

impl std::fmt::Display for G2pFfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for G2pFfiError {}

impl From<ArtifactError> for G2pFfiError {
    fn from(err: ArtifactError) -> Self {
        G2pFfiError::new(G2pErrorKind::Artifact, err.to_string())
    }
}

impl From<PipelineError> for G2pFfiError {
    fn from(err: PipelineError) -> Self {
        match err {
            PipelineError::Tokenizer(inner) => G2pFfiError::new(G2pErrorKind::Tokenizer, inner.to_string()),
            PipelineError::Core(inner) => G2pFfiError::new(G2pErrorKind::Config, inner.to_string()),
            PipelineError::Tvm(inner) => G2pFfiError::new(G2pErrorKind::Tvm, inner.to_string()),
            PipelineError::BatchSizeOverflow { .. } => G2pFfiError::new(G2pErrorKind::Config, err.to_string()),
            PipelineError::MissingEosToken => G2pFfiError::new(G2pErrorKind::Tokenizer, err.to_string()),
            PipelineError::MissingKvArtifacts => G2pFfiError::new(G2pErrorKind::Artifact, err.to_string()),
        }
    }
}

impl From<TokenizerError> for G2pFfiError {
    fn from(err: TokenizerError) -> Self {
        G2pFfiError::new(G2pErrorKind::Tokenizer, err.to_string())
    }
}

impl From<TvmError> for G2pFfiError {
    fn from(err: TvmError) -> Self {
        G2pFfiError::new(G2pErrorKind::Tvm, err.to_string())
    }
}

#[frb(opaque)]
pub struct G2pModel {
    pipeline: Mutex<G2pPipeline>,
}

#[frb]
pub fn g2p_platform_defaults() -> G2pPlatformDefaults {
    let target = default_target_for_platform().to_string();
    let device = default_device_for_target(&target).to_string();
    G2pPlatformDefaults { target, device }
}

#[frb]
pub fn g2p_default_config() -> G2pModelConfig {
    G2pModelConfig::default()
}

#[frb]
pub fn g2p_model_new(config: G2pModelConfig) -> Result<G2pModel, G2pFfiError> {
    let asset_root = normalize_non_empty("asset_root", &config.asset_root)?;
    let checkpoint = normalize_non_empty("checkpoint", &config.checkpoint)?;
    let target = if config.target.trim().is_empty() {
        default_target_for_platform().to_string()
    } else {
        config.target.trim().to_string()
    };

    let max_input_bytes = validate_positive("max_input_bytes", config.max_input_bytes)?;
    let max_output_len = validate_positive("max_output_len", config.max_output_len)?;
    let batch_size = validate_positive("batch_size", config.batch_size)?;

    let mut roots = ArtifactRoots::new(PathBuf::from(asset_root));
    if let Some(tokenizer_root) = config.tokenizer_root.as_ref().filter(|value| !value.is_empty()) {
        roots = roots.with_tokenizer_root(PathBuf::from(tokenizer_root));
    }
    if let Some(tvm_root) = config.tvm_root.as_ref().filter(|value| !value.is_empty()) {
        roots = roots.with_tvm_root(PathBuf::from(tvm_root));
    }
    let resolver = ArtifactResolver::with_roots(vec![roots]);
    let spec = ArtifactSpec {
        checkpoint: checkpoint.to_string(),
        max_input_bytes,
        max_output_len,
        batch_size,
        target: target.clone(),
    };
    let tokenizer_metadata_path = resolver.resolve_tokenizer_metadata(&spec)?;
    let metadata = load_tokenizer_metadata(&tokenizer_metadata_path)?;
    if !is_byt5_metadata(&metadata) {
        return Err(G2pFfiError::new(
            G2pErrorKind::Tokenizer,
            format!(
                "Unsupported tokenizer: {}. Only ByT5 byte tokenizers are supported.",
                metadata.tokenizer_name
            ),
        ));
    }
    let artifacts = resolver.resolve_tvm_artifacts(&spec, config.tvm_ext.as_deref(), config.use_kv_cache)?;

    let device_name = config
        .device
        .as_ref()
        .filter(|value| !value.is_empty())
        .cloned()
        .unwrap_or_else(|| default_device_for_target(&target).to_string());
    let device =
        DeviceConfig::from_str(&device_name, config.device_id).map_err(|err| {
            G2pFfiError::new(G2pErrorKind::Device, err.to_string())
        })?;

    let core_config = G2pConfig {
        max_input_bytes,
        max_output_len,
        space_after_colon: false,
    };
    let pipeline_config = PipelineConfig::from_core(&core_config, batch_size, None, config.use_kv_cache, device);
    let pipeline = G2pPipeline::load(tokenizer_metadata_path, artifacts, pipeline_config)?;

    Ok(G2pModel {
        pipeline: Mutex::new(pipeline),
    })
}

#[frb]
pub fn g2p_model_run(
    model: &G2pModel,
    words: Vec<String>,
    lang: String,
    options: Option<G2pRunOptions>,
) -> Result<Vec<String>, G2pFfiError> {
    let options = options.unwrap_or_default();
    let pipeline = model
        .pipeline
        .lock()
        .map_err(|_| G2pFfiError::new(G2pErrorKind::Inference, "Pipeline lock poisoned"))?;
    pipeline.run(&words, &lang, options.space_after_colon).map_err(G2pFfiError::from)
}

fn default_target_for_platform() -> &'static str {
    if cfg!(target_os = "ios") {
        "metal-ios"
    } else if cfg!(target_os = "android") {
        "vulkan"
    } else {
        "llvm"
    }
}

fn default_device_for_target(target: &str) -> &'static str {
    match target {
        "metal" | "metal-macos" | "metal-ios" => "metal",
        "cuda" => "cuda",
        "rocm" => "rocm",
        "vulkan" => "vulkan",
        "opencl" => "opencl",
        "webgpu" => "webgpu",
        _ => "cpu",
    }
}

fn validate_positive(label: &str, value: u32) -> Result<usize, G2pFfiError> {
    if value == 0 {
        return Err(G2pFfiError::new(
            G2pErrorKind::Config,
            format!("{label} must be >= 1"),
        ));
    }
    Ok(value as usize)
}

fn normalize_non_empty(label: &str, value: &str) -> Result<String, G2pFfiError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(G2pFfiError::new(
            G2pErrorKind::Config,
            format!("{label} is required"),
        ));
    }
    Ok(trimmed.to_string())
}

fn is_byt5_metadata(metadata: &TokenizerMetadata) -> bool {
    if metadata.byt5_offset.is_some() {
        return true;
    }
    metadata
        .tokenizer_name
        .to_ascii_lowercase()
        .contains("byt5")
}
