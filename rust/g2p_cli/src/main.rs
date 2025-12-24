use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::process;

use clap::Parser;

use charsiug2p_g2p_core::G2pConfig;
use charsiug2p_g2p_pipeline::{
    ArtifactResolver, ArtifactRoots, ArtifactSpec, G2pPipeline, PipelineConfig,
};
use charsiug2p_g2p_tvm::TvmArtifacts;

#[derive(Parser, Debug)]
#[command(name = "charsiug2p-g2p", about = "Run CharsiuG2P inference with TVM artifacts")]
struct Args {
    #[arg(long, help = "Path to tokenizer_metadata.json (auto-derived when omitted)")]
    tokenizer_metadata: Option<PathBuf>,
    #[arg(long, help = "Path to encoder artifact (.so/.dylib) (auto-derived when omitted)")]
    encoder: Option<PathBuf>,
    #[arg(long, help = "Path to decoder artifact (.so/.dylib) (auto-derived when omitted)")]
    decoder: Option<PathBuf>,
    #[arg(
        long,
        default_value = "charsiu/g2p_multilingual_byT5_tiny_8_layers_100",
        help = "Checkpoint name used for auto-derived paths"
    )]
    checkpoint: String,
    #[arg(
        long,
        default_value = "llvm",
        help = "TVM target directory name for auto-derived paths"
    )]
    tvm_target: String,
    #[arg(long, help = "TVM artifact extension (e.g., so, dylib, wasm)")]
    tvm_ext: Option<String>,
    #[arg(
        long,
        env = "CHARSIUG2P_ASSET_ROOT",
        help = "Root directory containing tokenizers/ and tvm/"
    )]
    artifact_root: Option<PathBuf>,
    #[arg(
        long,
        env = "CHARSIUG2P_TOKENIZER_ROOT",
        help = "Override tokenizer root (defaults to <artifact-root>/tokenizers)"
    )]
    tokenizer_root: Option<PathBuf>,
    #[arg(
        long,
        env = "CHARSIUG2P_TVM_ROOT",
        help = "Override TVM root (defaults to <artifact-root>/tvm)"
    )]
    tvm_root: Option<PathBuf>,
    #[arg(long, default_value = "eng-us", help = "Language code (e.g., eng-us)")]
    lang: String,
    #[arg(long, default_value_t = 64, help = "Max bytes for prefixed word")]
    max_input_bytes: usize,
    #[arg(long, default_value_t = 128, help = "Max output length")]
    max_output_len: usize,
    #[arg(long, default_value_t = 1, help = "Compiled batch size")]
    batch_size: usize,
    #[arg(long, default_value_t = false, help = "Insert a space after the language prefix")]
    space_after_colon: bool,
    #[arg(help = "Words to convert to phonemes")]
    words: Vec<String>,
}

fn main() {
    let args = Args::parse();
    if args.words.is_empty() {
        eprintln!("No words provided.");
        process::exit(1);
    }

    let core_config = G2pConfig {
        max_input_bytes: args.max_input_bytes,
        max_output_len: args.max_output_len,
        space_after_colon: args.space_after_colon,
    };
    let pipeline_config = PipelineConfig::from_core(&core_config, args.batch_size, None);
    let artifact_spec = ArtifactSpec {
        checkpoint: args.checkpoint.clone(),
        max_input_bytes: args.max_input_bytes,
        max_output_len: args.max_output_len,
        batch_size: args.batch_size,
        target: args.tvm_target.clone(),
    };
    let artifact_roots = build_artifact_roots(&args);
    let resolver = ArtifactResolver::with_roots(artifact_roots);
    let tokenizer_metadata = match args.tokenizer_metadata.clone() {
        Some(path) => path,
        None => match resolver.resolve_tokenizer_metadata(&artifact_spec) {
            Ok(path) => path,
            Err(err) => {
                eprintln!("Failed to locate tokenizer metadata: {err}");
                process::exit(1);
            }
        },
    };
    let artifacts = match (args.encoder.clone(), args.decoder.clone()) {
        (Some(encoder), Some(decoder)) => TvmArtifacts::new(encoder, decoder),
        (None, None) => match resolver.resolve_tvm_artifacts(&artifact_spec, args.tvm_ext.as_deref()) {
            Ok(artifacts) => artifacts,
            Err(err) => {
                eprintln!("Failed to locate TVM artifacts: {err}");
                process::exit(1);
            }
        },
        _ => {
            eprintln!("Both --encoder and --decoder must be provided together.");
            process::exit(1);
        }
    };
    let pipeline = match G2pPipeline::load(tokenizer_metadata, artifacts, pipeline_config) {
        Ok(pipeline) => pipeline,
        Err(err) => {
            eprintln!("Failed to initialize pipeline: {err}");
            process::exit(1);
        }
    };

    match pipeline.run(&args.words, &args.lang, args.space_after_colon) {
        Ok(outputs) => {
            for (word, phoneme) in args.words.iter().zip(outputs.iter()) {
                println!("{word}\t{phoneme}");
            }
        }
        Err(err) => {
            eprintln!("Inference failed: {err}");
            process::exit(1);
        }
    }
}

fn build_artifact_roots(args: &Args) -> Vec<ArtifactRoots> {
    if args.artifact_root.is_some() || args.tokenizer_root.is_some() || args.tvm_root.is_some() {
        let base = args.artifact_root.clone().unwrap_or_else(|| PathBuf::from("."));
        let mut root = ArtifactRoots::new(base);
        if let Some(tokenizer_root) = args.tokenizer_root.clone() {
            root = root.with_tokenizer_root(tokenizer_root);
        }
        if let Some(tvm_root) = args.tvm_root.clone() {
            root = root.with_tvm_root(tvm_root);
        }
        return vec![root];
    }

    let candidates = default_candidate_roots();
    let mut roots = candidates
        .iter()
        .filter(|path| path.exists())
        .cloned()
        .map(ArtifactRoots::new)
        .collect::<Vec<_>>();
    if roots.is_empty() {
        roots = candidates.into_iter().map(ArtifactRoots::new).collect();
    }
    roots
}

fn default_candidate_roots() -> Vec<PathBuf> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let candidates = vec![
        cwd.join("dist"),
        cwd.join("python").join("dist"),
        cwd.join("..").join("dist"),
        cwd.join("..").join("python").join("dist"),
        cwd.join("..").join("..").join("dist"),
        cwd.join("..").join("..").join("python").join("dist"),
    ];
    dedupe_paths(candidates)
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for path in paths {
        let key = path.to_string_lossy().to_string();
        if seen.insert(key) {
            result.push(path);
        }
    }
    result
}
