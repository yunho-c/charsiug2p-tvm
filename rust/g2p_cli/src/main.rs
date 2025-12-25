use std::collections::{HashMap, HashSet};
use std::env;
use std::path::PathBuf;
use std::process;

use clap::Parser;

use charsiug2p_g2p_core::G2pConfig;
use charsiug2p_g2p_pipeline::{
    ArtifactResolver, ArtifactRoots, ArtifactSpec, G2pPipeline, PipelineConfig,
};
use charsiug2p_g2p_tvm::{DeviceConfig, TvmArtifacts};

#[derive(Parser, Debug)]
#[command(name = "charsiug2p-g2p", about = "Run CharsiuG2P inference with TVM artifacts")]
struct Args {
    #[arg(long, help = "Path to tokenizer_metadata.json (auto-derived when omitted)")]
    tokenizer_metadata: Option<PathBuf>,
    #[arg(long, help = "Path to encoder artifact (.so/.dylib) (auto-derived when omitted)")]
    encoder: Option<PathBuf>,
    #[arg(long, help = "Path to decoder artifact (.so/.dylib) (auto-derived when omitted)")]
    decoder: Option<PathBuf>,
    #[arg(long, help = "Path to decoder_prefill artifact (.so/.dylib)")]
    decoder_prefill: Option<PathBuf>,
    #[arg(long, help = "Path to decoder_step artifact (.so/.dylib)")]
    decoder_step: Option<PathBuf>,
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
    #[arg(
        long,
        value_delimiter = ',',
        help = "Batch sizes to select from (comma-separated or repeatable)"
    )]
    batch_sizes: Vec<usize>,
    #[arg(
        long,
        default_value_t = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set,
        value_parser = clap::builder::BoolishValueParser::new(),
        help = "Enable KV-cache artifacts (use --kv-cache=false for cacheless decode)"
    )]
    kv_cache: bool,
    #[arg(long, help = "Device (cpu, cuda, metal, vulkan, opencl, webgpu, rocm)")]
    device: Option<String>,
    #[arg(long, default_value_t = 0, help = "Device id")]
    device_id: i32,
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
    let resolved_device = args
        .device
        .clone()
        .unwrap_or_else(|| default_device_for_target(&args.tvm_target).to_string());
    let device = match DeviceConfig::from_str(&resolved_device, args.device_id) {
        Ok(device) => device,
        Err(err) => {
            eprintln!("Invalid device: {err}");
            process::exit(1);
        }
    };
    let pipeline_config = PipelineConfig::from_core(
        &core_config,
        args.batch_size,
        None,
        args.kv_cache,
        device,
    );
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
    if !args.batch_sizes.is_empty()
        && (args.encoder.is_some()
            || args.decoder.is_some()
            || args.decoder_prefill.is_some()
            || args.decoder_step.is_some())
    {
        eprintln!("--batch-sizes cannot be used with explicit artifact paths.");
        process::exit(1);
    }

    if !args.batch_sizes.is_empty() {
        let batch_sizes = normalize_batch_sizes(&args.batch_sizes).unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });
        let outputs = run_with_batch_sizes(
            &args.words,
            &args.lang,
            args.space_after_colon,
            &tokenizer_metadata,
            &artifact_spec,
            &resolver,
            &core_config,
            args.kv_cache,
            device,
            args.tvm_ext.as_deref(),
            &batch_sizes,
        )
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });
        for (word, phoneme) in args.words.iter().zip(outputs.iter()) {
            println!("{word}\t{phoneme}");
        }
        return;
    }

    let artifacts = match (args.encoder.clone(), args.decoder.clone()) {
        (Some(encoder), Some(decoder)) => {
            let mut artifacts = TvmArtifacts::new(encoder, decoder);
            if args.kv_cache {
                let (prefill, step) = resolve_cache_paths(&artifacts, &args).unwrap_or_else(|err| {
                    eprintln!("{err}");
                    process::exit(1);
                });
                artifacts = artifacts.with_cache(prefill, step);
            }
            artifacts
        }
        (None, None) => match resolver.resolve_tvm_artifacts(
            &artifact_spec,
            args.tvm_ext.as_deref(),
            args.kv_cache,
        ) {
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

fn resolve_cache_paths(artifacts: &TvmArtifacts, args: &Args) -> Result<(PathBuf, PathBuf), String> {
    match (args.decoder_prefill.clone(), args.decoder_step.clone()) {
        (Some(prefill), Some(step)) => return Ok((prefill, step)),
        (Some(_), None) | (None, Some(_)) => {
            return Err("Both --decoder-prefill and --decoder-step must be provided together.".to_string())
        }
        (None, None) => {}
    }

    let dir = artifacts
        .encoder
        .parent()
        .or_else(|| artifacts.decoder.parent())
        .map(|path| path.to_path_buf())
        .ok_or_else(|| "Unable to infer decoder_prefill/decoder_step directory.".to_string())?;
    let ext = artifacts
        .encoder
        .extension()
        .or_else(|| artifacts.decoder.extension())
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| "Unable to infer decoder_prefill/decoder_step extension.".to_string())?;
    let prefill = dir.join(format!("decoder_prefill.{ext}"));
    let step = dir.join(format!("decoder_step.{ext}"));
    if prefill.exists() && step.exists() {
        Ok((prefill, step))
    } else {
        Err(format!(
            "KV-cache enabled but decoder_prefill/decoder_step not found at {} or {}.",
            prefill.display(),
            step.display()
        ))
    }
}

fn normalize_batch_sizes(values: &[usize]) -> Result<Vec<usize>, String> {
    let mut sizes = values.to_vec();
    if sizes.is_empty() {
        return Ok(sizes);
    }
    if sizes.iter().any(|size| *size == 0) {
        return Err("Batch sizes must be positive.".to_string());
    }
    sizes.sort_unstable();
    sizes.dedup();
    Ok(sizes)
}

fn pick_batch_size(available: &[usize], needed: usize) -> Result<usize, String> {
    if available.is_empty() {
        return Err("No batch sizes provided.".to_string());
    }
    if available[0] == 0 {
        return Err("Batch sizes must be positive.".to_string());
    }
    if needed <= available[0] {
        return Ok(available[0]);
    }
    for size in available {
        if *size >= needed {
            return Ok(*size);
        }
    }
    Ok(*available.last().unwrap())
}

fn run_with_batch_sizes(
    words: &[String],
    lang: &str,
    space_after_colon: bool,
    tokenizer_metadata: &PathBuf,
    artifact_spec: &ArtifactSpec,
    resolver: &ArtifactResolver,
    core_config: &G2pConfig,
    kv_cache: bool,
    device: DeviceConfig,
    tvm_ext: Option<&str>,
    batch_sizes: &[usize],
) -> Result<Vec<String>, String> {
    let mut pipelines = HashMap::<usize, G2pPipeline>::new();
    let mut results = Vec::with_capacity(words.len());
    let mut index = 0usize;
    while index < words.len() {
        let remaining = words.len() - index;
        let batch_size = pick_batch_size(batch_sizes, remaining)?;
        if !pipelines.contains_key(&batch_size) {
            let spec = ArtifactSpec {
                batch_size,
                ..artifact_spec.clone()
            };
            let artifacts = resolver
                .resolve_tvm_artifacts(&spec, tvm_ext, kv_cache)
                .map_err(|err| format!("Failed to locate TVM artifacts for batch_size={batch_size}: {err}"))?;
            let config = PipelineConfig::from_core(core_config, batch_size, None, kv_cache, device);
            let pipeline = G2pPipeline::load(tokenizer_metadata, artifacts, config)
                .map_err(|err| format!("Failed to initialize pipeline for batch_size={batch_size}: {err}"))?;
            pipelines.insert(batch_size, pipeline);
        }
        let chunk_len = remaining.min(batch_size);
        let chunk = words[index..index + chunk_len].to_vec();
        let pipeline = pipelines.get(&batch_size).expect("pipeline missing");
        let outputs = pipeline
            .run(&chunk, lang, space_after_colon)
            .map_err(|err| format!("Inference failed for batch_size={batch_size}: {err}"))?;
        results.extend(outputs);
        index += chunk_len;
    }
    Ok(results)
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
