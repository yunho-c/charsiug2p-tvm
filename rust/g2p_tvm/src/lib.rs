use std::fmt;
use std::path::{Path, PathBuf};

use tvm_ffi::{AnyView, Function, Module, Tensor};

pub const MAIN_FUNCTION: &str = "main";
const VM_LOAD_FUNCTION: &str = "vm_load_executable";
const VM_INIT_FUNCTION: &str = "vm_initialization";
const DL_DEVICE_TYPE_CPU: i32 = 1;
const DEFAULT_DEVICE_ID: i32 = 0;
const POOLED_ALLOCATOR: i32 = 2;

#[derive(Debug)]
pub enum TvmError {
    MissingArtifact(PathBuf),
    MissingFunction { module: PathBuf, name: String },
    MissingVmLoader { module: PathBuf },
    MissingVmInitialization { module: PathBuf },
    MissingVmEntry { module: PathBuf, name: String },
    UnexpectedOutputType(i32),
    Ffi(tvm_ffi::Error),
}

impl fmt::Display for TvmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TvmError::MissingArtifact(path) => write!(f, "Missing artifact: {}", path.display()),
            TvmError::MissingFunction { module, name } => write!(
                f,
                "Function '{name}' not found in module {}. If this is a Relax VM executable, use vm_load_executable + vm_initialization and fetch '{name}' from the VM module.",
                module.display()
            ),
            TvmError::MissingVmLoader { module } => write!(
                f,
                "Relax VM loader '{VM_LOAD_FUNCTION}' not found in module {}.",
                module.display()
            ),
            TvmError::MissingVmInitialization { module } => write!(
                f,
                "Relax VM initialization function '{VM_INIT_FUNCTION}' not found in module {}.",
                module.display()
            ),
            TvmError::MissingVmEntry { module, name } => write!(
                f,
                "Function '{name}' not found in Relax VM executable for module {}. Check the entry name (often 'main').",
                module.display()
            ),
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
    path: PathBuf,
}

impl TvmModule {
    pub fn load(path: &Path) -> Result<Self, TvmError> {
        let module = Module::load_from_file(path.to_string_lossy().as_ref())?;
        Ok(Self {
            module,
            path: path.to_path_buf(),
        })
    }

    pub fn main(&self) -> Result<Function, TvmError> {
        self.module.get_function(MAIN_FUNCTION).map_err(|_error| {
            TvmError::MissingFunction {
                module: self.path.clone(),
                name: MAIN_FUNCTION.to_string(),
            }
        })
    }

    fn load_entry(self, name: &str, vm_config: VmConfig) -> Result<LoadedModule, TvmError> {
        if let Ok(function) = self.module.get_function(name) {
            return Ok(LoadedModule {
                library: self.module,
                executable: None,
                entry: function,
            });
        }

        let vm_load = self
            .module
            .get_function(VM_LOAD_FUNCTION)
            .map_err(|_error| TvmError::MissingVmLoader {
                module: self.path.clone(),
            })?;
        let exec_any = vm_load.call_tuple(())?;
        let exec_module: Module = exec_any.try_into()?;
        let vm_init = exec_module
            .get_function(VM_INIT_FUNCTION)
            .map_err(|_error| TvmError::MissingVmInitialization {
                module: self.path.clone(),
            })?;
        initialize_vm(&vm_init, vm_config)?;
        let entry = exec_module.get_function(name).map_err(|_error| TvmError::MissingVmEntry {
            module: self.path.clone(),
            name: name.to_string(),
        })?;
        Ok(LoadedModule {
            library: self.module,
            executable: Some(exec_module),
            entry,
        })
    }
}

pub struct TvmExecutable {
    encoder: LoadedModule,
    decoder: LoadedModule,
}

impl TvmExecutable {
    pub fn load(artifacts: &TvmArtifacts) -> Result<Self, TvmError> {
        artifacts.validate()?;
        let vm_config = VmConfig::cpu();
        let encoder = TvmModule::load(&artifacts.encoder)?.load_entry(MAIN_FUNCTION, vm_config)?;
        let decoder = TvmModule::load(&artifacts.decoder)?.load_entry(MAIN_FUNCTION, vm_config)?;
        Ok(Self {
            encoder,
            decoder,
        })
    }

    pub fn encoder_main(&self) -> &Function {
        &self.encoder.entry
    }

    pub fn decoder_main(&self) -> &Function {
        &self.decoder.entry
    }

    pub fn call_encoder(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor, TvmError> {
        let args = [AnyView::from(input_ids), AnyView::from(attention_mask)];
        self.call_tensor(&self.encoder.entry, &args)
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
        self.call_tensor(&self.decoder.entry, &args)
    }

    fn call_tensor(&self, function: &Function, args: &[AnyView<'_>]) -> Result<Tensor, TvmError> {
        let output = function.call_packed(args)?;
        extract_tensor_from_output(&output, 0)
    }
}

#[derive(Clone, Copy)]
struct VmConfig {
    device_type: i32,
    device_id: i32,
}

impl VmConfig {
    fn cpu() -> Self {
        Self {
            device_type: DL_DEVICE_TYPE_CPU,
            device_id: DEFAULT_DEVICE_ID,
        }
    }
}

struct LoadedModule {
    #[allow(dead_code)]
    library: Module,
    #[allow(dead_code)]
    executable: Option<Module>,
    entry: Function,
}

fn initialize_vm(vm_init: &Function, config: VmConfig) -> Result<(), TvmError> {
    if config.device_type != DL_DEVICE_TYPE_CPU {
        vm_init.call_tuple((
            config.device_type,
            config.device_id,
            POOLED_ALLOCATOR,
            DL_DEVICE_TYPE_CPU,
            DEFAULT_DEVICE_ID,
            POOLED_ALLOCATOR,
        ))?;
    } else {
        vm_init.call_tuple((config.device_type, config.device_id, POOLED_ALLOCATOR))?;
    }
    Ok(())
}

fn extract_tensor_from_output(output: &tvm_ffi::Any, index: usize) -> Result<Tensor, TvmError> {
    if let Some(tensor) = output.try_as::<Tensor>() {
        return Ok(tensor);
    }
    if output.type_index() == tvm_ffi::TypeIndex::kTVMFFIArray as i32 {
        let array_get_item = Function::get_global("ffi.ArrayGetItem")?;
        let index_val = index as i64;
        let args = [AnyView::from(output), AnyView::from(&index_val)];
        let element = array_get_item.call_packed(&args)?;
        return element.try_as::<Tensor>().ok_or(TvmError::UnexpectedOutputType(element.type_index()));
    }
    Err(TvmError::UnexpectedOutputType(output.type_index()))
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
