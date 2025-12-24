use std::path::PathBuf;
use std::process;

use clap::Parser;

use charsiug2p_g2p_core::G2pConfig;
use charsiug2p_g2p_pipeline::{G2pPipeline, PipelineConfig};
use charsiug2p_g2p_tvm::TvmArtifacts;

#[derive(Parser, Debug)]
#[command(name = "charsiug2p-g2p", about = "Run CharsiuG2P inference with TVM artifacts")]
struct Args {
    #[arg(long, help = "Path to tokenizer_metadata.json")]
    tokenizer_metadata: PathBuf,
    #[arg(long, help = "Path to encoder artifact (.so/.dylib)")]
    encoder: PathBuf,
    #[arg(long, help = "Path to decoder artifact (.so/.dylib)")]
    decoder: PathBuf,
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
    let artifacts = TvmArtifacts::new(args.encoder, args.decoder);
    let pipeline = match G2pPipeline::load(args.tokenizer_metadata, artifacts, pipeline_config) {
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
