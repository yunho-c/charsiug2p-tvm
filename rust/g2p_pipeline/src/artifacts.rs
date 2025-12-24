use std::fmt;
use std::path::{Path, PathBuf};

use charsiug2p_g2p_tvm::TvmArtifacts;

#[derive(Debug, Clone)]
pub struct ArtifactSpec {
    pub checkpoint: String,
    pub max_input_bytes: usize,
    pub max_output_len: usize,
    pub batch_size: usize,
    pub target: String,
}

impl ArtifactSpec {
    pub fn safe_checkpoint(&self) -> String {
        self.checkpoint.replace("/", "_")
    }

    pub fn tokenizer_subdir(&self) -> String {
        format!("in{}_out{}", self.max_input_bytes, self.max_output_len)
    }

    pub fn tvm_subdir(&self) -> String {
        format!(
            "b{}_in{}_out{}",
            self.batch_size, self.max_input_bytes, self.max_output_len
        )
    }
}

#[derive(Debug, Clone)]
pub struct ArtifactRoots {
    base: PathBuf,
    tokenizer_root: Option<PathBuf>,
    tvm_root: Option<PathBuf>,
}

impl ArtifactRoots {
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self {
            base: base.into(),
            tokenizer_root: None,
            tvm_root: None,
        }
    }

    pub fn with_tokenizer_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.tokenizer_root = Some(root.into());
        self
    }

    pub fn with_tvm_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.tvm_root = Some(root.into());
        self
    }

    pub fn tokenizers_dir(&self) -> PathBuf {
        self.tokenizer_root.clone().unwrap_or_else(|| self.base.join("tokenizers"))
    }

    pub fn tvm_dir(&self) -> PathBuf {
        self.tvm_root.clone().unwrap_or_else(|| self.base.join("tvm"))
    }
}

#[derive(Debug)]
pub enum ArtifactError {
    MissingTokenizerMetadata { candidates: Vec<PathBuf> },
    MissingTvmArtifacts {
        encoder_candidates: Vec<PathBuf>,
        decoder_candidates: Vec<PathBuf>,
    },
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArtifactError::MissingTokenizerMetadata { candidates } => {
                write!(
                    f,
                    "Tokenizer metadata not found. Tried: {}",
                    format_candidates(candidates)
                )
            }
            ArtifactError::MissingTvmArtifacts {
                encoder_candidates,
                decoder_candidates,
            } => {
                write!(
                    f,
                    "TVM artifacts not found. Encoder tried: {}. Decoder tried: {}",
                    format_candidates(encoder_candidates),
                    format_candidates(decoder_candidates)
                )
            }
        }
    }
}

impl std::error::Error for ArtifactError {}

#[derive(Debug, Clone)]
pub struct ArtifactResolver {
    roots: Vec<ArtifactRoots>,
}

impl ArtifactResolver {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            roots: vec![ArtifactRoots::new(root)],
        }
    }

    pub fn with_roots(roots: Vec<ArtifactRoots>) -> Self {
        Self { roots }
    }

    pub fn resolve_tokenizer_metadata(&self, spec: &ArtifactSpec) -> Result<PathBuf, ArtifactError> {
        let safe_checkpoint = spec.safe_checkpoint();
        let subdir = spec.tokenizer_subdir();
        let mut candidates = Vec::new();
        for root in &self.roots {
            let path = root
                .tokenizers_dir()
                .join(&safe_checkpoint)
                .join(&subdir)
                .join("tokenizer_metadata.json");
            if path.exists() {
                return Ok(path);
            }
            candidates.push(path);
        }
        Err(ArtifactError::MissingTokenizerMetadata { candidates })
    }

    pub fn resolve_tvm_artifacts(
        &self,
        spec: &ArtifactSpec,
        ext_hint: Option<&str>,
    ) -> Result<TvmArtifacts, ArtifactError> {
        let safe_checkpoint = spec.safe_checkpoint();
        let subdir = spec.tvm_subdir();
        let mut encoder_candidates = Vec::new();
        let mut decoder_candidates = Vec::new();
        for root in &self.roots {
            let target_dir = root
                .tvm_dir()
                .join(&safe_checkpoint)
                .join(&subdir)
                .join(&spec.target);
            let encoder_paths = module_candidates(&target_dir, "encoder", ext_hint);
            let decoder_paths = module_candidates(&target_dir, "decoder", ext_hint);
            let encoder = first_existing(&encoder_paths);
            let decoder = first_existing(&decoder_paths);
            if let (Some(encoder), Some(decoder)) = (encoder, decoder) {
                return Ok(TvmArtifacts::new(encoder, decoder));
            }
            encoder_candidates.extend(encoder_paths);
            decoder_candidates.extend(decoder_paths);
        }
        Err(ArtifactError::MissingTvmArtifacts {
            encoder_candidates,
            decoder_candidates,
        })
    }
}

fn module_candidates(dir: &Path, stem: &str, ext_hint: Option<&str>) -> Vec<PathBuf> {
    let mut exts = Vec::new();
    if let Some(ext) = ext_hint.and_then(normalize_extension) {
        exts.push(ext);
    }
    for ext in ["so", "dylib", "wasm"] {
        if !exts.iter().any(|value| value == ext) {
            exts.push(ext.to_string());
        }
    }
    exts.into_iter()
        .map(|ext| dir.join(format!("{stem}.{ext}")))
        .collect()
}

fn normalize_extension(ext: &str) -> Option<String> {
    let trimmed = ext.trim().trim_start_matches('.');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|path| path.exists()).cloned()
}

fn format_candidates(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        return "(none)".to_string();
    }
    let limit = 5usize;
    let shown = paths.iter().take(limit).map(|path| path.display().to_string()).collect::<Vec<_>>();
    if paths.len() > limit {
        format!("{} (+{} more)", shown.join(", "), paths.len() - limit)
    } else {
        shown.join(", ")
    }
}
