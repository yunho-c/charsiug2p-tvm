use charsiug2p_g2p_tvm::{DeviceConfig, LoadedModule, TvmModule, MAIN_FUNCTION, tensor_to_vec_f32};
use tvm_ffi::Tensor;

#[test]
fn system_lib_smoke() {
    let prefix = match std::env::var("TVM_SYSTEM_LIB_PREFIX") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => {
            eprintln!("skipping: set TVM_SYSTEM_LIB_PREFIX to run the system-lib test");
            return;
        }
    };

    let module = TvmModule::from_system_lib(prefix.trim()).expect("system lib not available");
    let loaded: LoadedModule = module
        .load_entry(MAIN_FUNCTION, DeviceConfig::cpu())
        .expect("failed to load entry");
    let input = Tensor::from_slice(&[1.5f32], &[1]).expect("input tensor");
    let output_any = loaded.entry().call_tuple((input,)).expect("call main");
    let output: Tensor = output_any.try_into().expect("output tensor");
    let values = tensor_to_vec_f32(&output).expect("output data");
    assert_eq!(values, vec![2.5f32]);
}
