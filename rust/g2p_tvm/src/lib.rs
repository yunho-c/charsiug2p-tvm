use std::ffi::c_void;
use std::fmt;
use std::mem;
use std::path::{Path, PathBuf};

use tvm_ffi::{
    AnyView, DLDataType, DLDataTypeCode, DLDataTypeExt, DLDevice, DLDeviceType, Function, Module,
    Shape, Tensor,
};

pub const MAIN_FUNCTION: &str = "main";
const VM_LOAD_FUNCTION: &str = "vm_load_executable";
const VM_INIT_FUNCTION: &str = "vm_initialization";
const DEFAULT_DEVICE_ID: i32 = 0;
const POOLED_ALLOCATOR: i32 = 2;

#[derive(Debug)]
pub enum TvmError {
    MissingArtifact(PathBuf),
    MissingFunction { module: PathBuf, name: String },
    MissingVmLoader { module: PathBuf },
    MissingVmInitialization { module: PathBuf },
    MissingVmEntry { module: PathBuf, name: String },
    MissingKvArtifacts {
        decoder_prefill: Option<PathBuf>,
        decoder_step: Option<PathBuf>,
    },
    InvalidDevice(String),
    TensorShapeMismatch { expected: usize, got: usize },
    InvalidShapeDimension(i64),
    TensorDtypeMismatch { expected: DLDataType, got: DLDataType },
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
            TvmError::MissingKvArtifacts {
                decoder_prefill,
                decoder_step,
            } => write!(
                f,
                "KV-cache artifacts are incomplete. decoder_prefill={:?}, decoder_step={:?}",
                decoder_prefill, decoder_step
            ),
            TvmError::InvalidDevice(device) => {
                write!(f, "Unsupported device '{device}' for TVM runtime.")
            }
            TvmError::TensorShapeMismatch { expected, got } => {
                write!(f, "Tensor shape mismatch: expected {expected}, got {got}")
            }
            TvmError::InvalidShapeDimension(value) => {
                write!(f, "Invalid shape dimension: {value}")
            }
            TvmError::TensorDtypeMismatch { expected, got } => {
                write!(
                    f,
                    "Tensor dtype mismatch: expected {}, got {}",
                    expected.to_string(),
                    got.to_string()
                )
            }
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

#[derive(Clone, Copy, Debug)]
pub struct DeviceConfig {
    device: DLDevice,
}

impl DeviceConfig {
    pub fn cpu() -> Self {
        Self {
            device: DLDevice::new(DLDeviceType::kDLCPU, DEFAULT_DEVICE_ID),
        }
    }

    pub fn from_str(device: &str, device_id: i32) -> Result<Self, TvmError> {
        let device_type = match device {
            "cpu" | "llvm" => DLDeviceType::kDLCPU,
            "cuda" | "gpu" => DLDeviceType::kDLCUDA,
            "metal" => DLDeviceType::kDLMetal,
            "vulkan" => DLDeviceType::kDLVulkan,
            "opencl" => DLDeviceType::kDLOpenCL,
            "webgpu" => DLDeviceType::kDLWebGPU,
            "rocm" => DLDeviceType::kDLROCM,
            _ => return Err(TvmError::InvalidDevice(device.to_string())),
        };
        Ok(Self {
            device: DLDevice::new(device_type, device_id),
        })
    }

    pub fn device(&self) -> DLDevice {
        self.device
    }

    pub fn device_type(&self) -> DLDeviceType {
        self.device.device_type
    }

    pub fn device_id(&self) -> i32 {
        self.device.device_id
    }

    pub fn is_cpu(&self) -> bool {
        self.device.device_type == DLDeviceType::kDLCPU
    }
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

    pub fn from_module(module: Module, label: impl Into<PathBuf>) -> Self {
        Self {
            module,
            path: label.into(),
        }
    }

    pub fn from_system_lib(prefix: &str) -> Result<Self, TvmError> {
        let system_lib = Function::get_global("ffi.SystemLib")?;
        let module_any = if prefix.is_empty() {
            system_lib.call_tuple(())?
        } else {
            let prefix_arg = tvm_ffi::String::from(prefix);
            system_lib.call_tuple((prefix_arg,))?
        };
        let module: Module = module_any.try_into()?;
        Ok(Self::from_module(
            module,
            PathBuf::from(format!("<system-lib:{prefix}>")),
        ))
    }

    pub fn main(&self) -> Result<Function, TvmError> {
        self.module.get_function(MAIN_FUNCTION).map_err(|_error| {
            TvmError::MissingFunction {
                module: self.path.clone(),
                name: MAIN_FUNCTION.to_string(),
            }
        })
    }

    pub fn load_entry(self, name: &str, device: DeviceConfig) -> Result<LoadedModule, TvmError> {
        let vm_config = VmConfig::new(device);
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
    decoder_prefill: Option<LoadedModule>,
    decoder_step: Option<LoadedModule>,
}

impl TvmExecutable {
    pub fn load(artifacts: &TvmArtifacts) -> Result<Self, TvmError> {
        Self::load_with_device(artifacts, DeviceConfig::cpu())
    }

    pub fn load_with_device(
        artifacts: &TvmArtifacts,
        device: DeviceConfig,
    ) -> Result<Self, TvmError> {
        artifacts.validate()?;
        let encoder = TvmModule::load(&artifacts.encoder)?.load_entry(MAIN_FUNCTION, device)?;
        let decoder = TvmModule::load(&artifacts.decoder)?.load_entry(MAIN_FUNCTION, device)?;
        let (decoder_prefill, decoder_step) = match (&artifacts.decoder_prefill, &artifacts.decoder_step) {
            (Some(prefill_path), Some(step_path)) => (
                Some(TvmModule::load(prefill_path)?.load_entry(MAIN_FUNCTION, device)?),
                Some(TvmModule::load(step_path)?.load_entry(MAIN_FUNCTION, device)?),
            ),
            (None, None) => (None, None),
            _ => {
                return Err(TvmError::MissingKvArtifacts {
                    decoder_prefill: artifacts.decoder_prefill.clone(),
                    decoder_step: artifacts.decoder_step.clone(),
                })
            }
        };
        Ok(Self {
            encoder,
            decoder,
            decoder_prefill,
            decoder_step,
        })
    }

    pub fn encoder_main(&self) -> &Function {
        &self.encoder.entry
    }

    pub fn decoder_main(&self) -> &Function {
        &self.decoder.entry
    }

    pub fn has_kv_cache(&self) -> bool {
        self.decoder_prefill.is_some() && self.decoder_step.is_some()
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

    pub fn call_decoder_prefill(
        &self,
        decoder_input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), TvmError> {
        let module = self.decoder_prefill.as_ref().ok_or(TvmError::MissingKvArtifacts {
            decoder_prefill: None,
            decoder_step: None,
        })?;
        let args = [
            AnyView::from(decoder_input_ids),
            AnyView::from(encoder_hidden_states),
            AnyView::from(encoder_attention_mask),
        ];
        let output = module.entry.call_packed(&args)?;
        let tensors = extract_tensor_array(&output, 4)?;
        let mut iter = tensors.into_iter();
        let logits = iter.next().expect("logits");
        let past_k = iter.next().expect("past_k");
        let past_v = iter.next().expect("past_v");
        let cur_pos = iter.next().expect("cur_pos");
        Ok((logits, past_k, past_v, cur_pos))
    }

    pub fn call_decoder_step(
        &self,
        decoder_input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
        past_k: &Tensor,
        past_v: &Tensor,
        cur_pos: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), TvmError> {
        let module = self.decoder_step.as_ref().ok_or(TvmError::MissingKvArtifacts {
            decoder_prefill: None,
            decoder_step: None,
        })?;
        let args = [
            AnyView::from(decoder_input_ids),
            AnyView::from(encoder_hidden_states),
            AnyView::from(encoder_attention_mask),
            AnyView::from(past_k),
            AnyView::from(past_v),
            AnyView::from(cur_pos),
        ];
        let output = module.entry.call_packed(&args)?;
        let tensors = extract_tensor_array(&output, 4)?;
        let mut iter = tensors.into_iter();
        let logits = iter.next().expect("logits");
        let next_k = iter.next().expect("next_k");
        let next_v = iter.next().expect("next_v");
        let next_pos = iter.next().expect("next_pos");
        Ok((logits, next_k, next_v, next_pos))
    }

    fn call_tensor(&self, function: &Function, args: &[AnyView<'_>]) -> Result<Tensor, TvmError> {
        let output = function.call_packed(args)?;
        extract_tensor_from_output(&output, 0)
    }
}

#[derive(Clone, Copy)]
struct VmConfig {
    device: DeviceConfig,
}

impl VmConfig {
    fn new(device: DeviceConfig) -> Self {
        Self { device }
    }
}

pub struct LoadedModule {
    #[allow(dead_code)]
    library: Module,
    #[allow(dead_code)]
    executable: Option<Module>,
    entry: Function,
}

impl LoadedModule {
    pub fn entry(&self) -> &Function {
        &self.entry
    }
}

fn initialize_vm(vm_init: &Function, config: VmConfig) -> Result<(), TvmError> {
    let device_type = config.device.device_type() as i32;
    let device_id = config.device.device_id();
    if config.device.device_type() != DLDeviceType::kDLCPU {
        vm_init.call_tuple((
            device_type,
            device_id,
            POOLED_ALLOCATOR,
            DLDeviceType::kDLCPU as i32,
            DEFAULT_DEVICE_ID,
            POOLED_ALLOCATOR,
        ))?;
    } else {
        vm_init.call_tuple((device_type, device_id, POOLED_ALLOCATOR))?;
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

fn extract_tensor_array(output: &tvm_ffi::Any, expected: usize) -> Result<Vec<Tensor>, TvmError> {
    if expected == 1 {
        return Ok(vec![extract_tensor_from_output(output, 0)?]);
    }
    if output.type_index() != tvm_ffi::TypeIndex::kTVMFFIArray as i32 {
        return Err(TvmError::UnexpectedOutputType(output.type_index()));
    }
    let array_get_item = Function::get_global("ffi.ArrayGetItem")?;
    let mut tensors = Vec::with_capacity(expected);
    for index in 0..expected {
        let index_val = index as i64;
        let args = [AnyView::from(output), AnyView::from(&index_val)];
        let element = array_get_item.call_packed(&args)?;
        let tensor = element
            .try_as::<Tensor>()
            .ok_or(TvmError::UnexpectedOutputType(element.type_index()))?;
        tensors.push(tensor);
    }
    Ok(tensors)
}

pub fn tensor_from_i64(data: &[i64], shape: &[i64]) -> Result<Tensor, TvmError> {
    tensor_from_i64_device(data, shape, &DeviceConfig::cpu())
}

pub fn tensor_from_i32(data: &[i32], shape: &[i64]) -> Result<Tensor, TvmError> {
    tensor_from_i32_device(data, shape, &DeviceConfig::cpu())
}

pub fn tensor_from_i64_device(
    data: &[i64],
    shape: &[i64],
    device: &DeviceConfig,
) -> Result<Tensor, TvmError> {
    tensor_from_slice_device(data, shape, device)
}

pub fn tensor_from_i32_device(
    data: &[i32],
    shape: &[i64],
    device: &DeviceConfig,
) -> Result<Tensor, TvmError> {
    tensor_from_slice_device(data, shape, device)
}

pub fn tensor_to_vec_f32(tensor: &Tensor) -> Result<Vec<f32>, TvmError> {
    let expected = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    if tensor.dtype() != expected {
        return Err(TvmError::TensorDtypeMismatch {
            expected,
            got: tensor.dtype(),
        });
    }
    if tensor.device().device_type == DLDeviceType::kDLCPU {
        return Ok(tensor.data_as_slice::<f32>()?.to_vec());
    }
    let numel = tensor.numel();
    let mut data = vec![0f32; numel];
    let nbytes = numel * mem::size_of::<f32>();
    tensor_copy_to_bytes(tensor, data.as_mut_ptr() as *mut c_void, nbytes)?;
    Ok(data)
}

pub fn any_from_tensor(tensor: &Tensor) -> AnyView<'_> {
    AnyView::from(tensor)
}

fn tensor_from_slice_device<T: tvm_ffi::dtype::AsDLDataType>(
    data: &[T],
    shape: &[i64],
    device: &DeviceConfig,
) -> Result<Tensor, TvmError> {
    let expected_len = shape_product(shape)?;
    if data.len() != expected_len {
        return Err(TvmError::TensorShapeMismatch {
            expected: expected_len,
            got: data.len(),
        });
    }
    if device.is_cpu() {
        return Ok(Tensor::from_slice(data, shape)?);
    }
    let dtype = T::DL_DATA_TYPE;
    let tensor = tensor_alloc(shape, dtype, device.device())?;
    let nbytes = data.len() * mem::size_of::<T>();
    tensor_copy_from_bytes(&tensor, data.as_ptr() as *mut c_void, nbytes)?;
    Ok(tensor)
}

fn tensor_alloc(shape: &[i64], dtype: DLDataType, device: DLDevice) -> Result<Tensor, TvmError> {
    let alloc = Function::get_global("runtime.TVMTensorAllocWithScope")?;
    let shape_obj = Shape::from(shape);
    let mem_scope: Option<tvm_ffi::String> = None;
    let alloc_args = [
        AnyView::from(&shape_obj),
        AnyView::from(&dtype),
        AnyView::from(&device),
        AnyView::from(&mem_scope),
    ];
    let tensor_any = alloc.call_packed(&alloc_args)?;
    tensor_any
        .try_as::<Tensor>()
        .ok_or(TvmError::UnexpectedOutputType(tensor_any.type_index()))
}

fn tensor_copy_from_bytes(
    tensor: &Tensor,
    data_ptr: *mut c_void,
    nbytes: usize,
) -> Result<(), TvmError> {
    let copy_from = Function::get_global("runtime.TVMTensorCopyFromBytes")?;
    let copy_args = [
        AnyView::from(tensor),
        AnyView::from(&data_ptr),
        AnyView::from(&nbytes),
    ];
    copy_from.call_packed(&copy_args)?;
    Ok(())
}

fn tensor_copy_to_bytes(tensor: &Tensor, data_ptr: *mut c_void, nbytes: usize) -> Result<(), TvmError> {
    let copy_to = Function::get_global("runtime.TVMTensorCopyToBytes")?;
    let copy_args = [
        AnyView::from(tensor),
        AnyView::from(&data_ptr),
        AnyView::from(&nbytes),
    ];
    copy_to.call_packed(&copy_args)?;
    Ok(())
}

fn shape_product(shape: &[i64]) -> Result<usize, TvmError> {
    let mut product = 1usize;
    for dim in shape {
        let dim_usize = usize::try_from(*dim).map_err(|_| TvmError::InvalidShapeDimension(*dim))?;
        product = product.saturating_mul(dim_usize);
    }
    Ok(product)
}
