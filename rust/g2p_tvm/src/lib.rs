use std::fmt;
use std::path::{Path, PathBuf};

use tvm_ffi::{AnyView, Function, Module, Tensor};

pub const MAIN_FUNCTION: &str = "main";

#[derive(Debug)]
pub enum TvmError {
    MissingArtifact(PathBuf),
    UnexpectedOutputType(i32),
    Ffi(tvm_ffi::Error),
}

impl fmt::Display for TvmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TvmError::MissingArtifact(path) => write!(f, "Missing artifact: {}", path.display()),
            TvmError::UnexpectedOutputType(type_index) => {
                write!(f, "Unexpected output type index: {type_index}")
            }
            TvmError::Ffi(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for TvmError {}

impl From<tvm_ffi::Error> for TvmError {
    fn from(error: tvm_ffi::Error) -> Self {
        TvmError::Ffi(error)
    }
}

#[derive(Debug, Clone)]
pub struct TvmArtifacts {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub decoder_prefill: Option<PathBuf>,
    pub decoder_step: Option<PathBuf>,
}

impl TvmArtifacts {
    pub fn new(encoder: impl Into<PathBuf>, decoder: impl Into<PathBuf>) -> Self {
        Self {
            encoder: encoder.into(),
            decoder: decoder.into(),
            decoder_prefill: None,
            decoder_step: None,
        }
    }

    pub fn with_cache(
        mut self,
        decoder_prefill: impl Into<PathBuf>,
        decoder_step: impl Into<PathBuf>,
    ) -> Self {
        self.decoder_prefill = Some(decoder_prefill.into());
        self.decoder_step = Some(decoder_step.into());
        self
    }

    pub fn validate(&self) -> Result<(), TvmError> {
        if !self.encoder.exists() {
            return Err(TvmError::MissingArtifact(self.encoder.clone()));
        }
        if !self.decoder.exists() {
            return Err(TvmError::MissingArtifact(self.decoder.clone()));
        }
        if let Some(path) = &self.decoder_prefill {
            if !path.exists() {
                return Err(TvmError::MissingArtifact(path.clone()));
            }
        }
        if let Some(path) = &self.decoder_step {
            if !path.exists() {
                return Err(TvmError::MissingArtifact(path.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct TvmModule {
    module: Module,
}

impl TvmModule {
    pub fn load(path: &Path) -> Result<Self, TvmError> {
        let module = Module::load_from_file(path.to_string_lossy().as_ref())?;
        Ok(Self { module })
    }

    pub fn main(&self) -> Result<Function, TvmError> {
        Ok(self.module.get_function(MAIN_FUNCTION)?)
    }
}

pub struct TvmExecutable {
    encoder: TvmModule,
    decoder: TvmModule,
    encoder_main: Function,
    decoder_main: Function,
}

impl TvmExecutable {
    pub fn load(artifacts: &TvmArtifacts) -> Result<Self, TvmError> {
        artifacts.validate()?;
        let encoder = TvmModule::load(&artifacts.encoder)?;
        let decoder = TvmModule::load(&artifacts.decoder)?;
        let encoder_main = encoder.main()?;
        let decoder_main = decoder.main()?;
        Ok(Self {
            encoder,
            decoder,
            encoder_main,
            decoder_main,
        })
    }

    pub fn encoder_main(&self) -> &Function {
        &self.encoder_main
    }

    pub fn decoder_main(&self) -> &Function {
        &self.decoder_main
    }

    pub fn call_encoder(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor, TvmError> {
        let args = [AnyView::from(input_ids), AnyView::from(attention_mask)];
        self.call_tensor(&self.encoder_main, &args)
    }

    pub fn call_decoder(
        &self,
        decoder_input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> Result<Tensor, TvmError> {
        let args = [
            AnyView::from(decoder_input_ids),
            AnyView::from(encoder_hidden_states),
            AnyView::from(encoder_attention_mask),
        ];
        self.call_tensor(&self.decoder_main, &args)
    }

    fn call_tensor(&self, function: &Function, args: &[AnyView<'_>]) -> Result<Tensor, TvmError> {
        // NOTE: This assumes the module exports MAIN_FUNCTION directly. Relax bytecode modules may need VM wiring.
        let output = function.call_packed(args)?;
        output.try_as::<Tensor>().ok_or(TvmError::UnexpectedOutputType(output.type_index()))
    }
}

pub fn tensor_from_i64(data: &[i64], shape: &[i64]) -> Result<Tensor, TvmError> {
    Ok(Tensor::from_slice(data, shape)?)
}

pub fn tensor_from_i32(data: &[i32], shape: &[i64]) -> Result<Tensor, TvmError> {
    Ok(Tensor::from_slice(data, shape)?)
}

pub fn any_from_tensor(tensor: &Tensor) -> AnyView<'_> {
    AnyView::from(tensor)
}
