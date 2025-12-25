use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=G2P_TVM_SYSTEM_LIB");
    println!("cargo:rerun-if-env-changed=G2P_TVM_RUNTIME_LIB");
    let output = Command::new("tvm-ffi-config")
        .arg("--libdir")
        .output()
        .expect("Failed to run tvm-ffi-config");
    if !output.status.success() {
        panic!("tvm-ffi-config failed with status {}", output.status);
    }
    let lib_dir = String::from_utf8(output.stdout)
        .expect("Invalid UTF-8 output from tvm-ffi-config")
        .trim()
        .to_string();
    if lib_dir.is_empty() {
        panic!("tvm-ffi-config returned an empty libdir");
    }
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=tvm_ffi");
    println!("cargo:rustc-link-lib=dylib=tvm_ffi_testing");
    let os_env_var = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "PATH",
        Ok("macos") => "DYLD_LIBRARY_PATH",
        Ok("linux") => "LD_LIBRARY_PATH",
        _ => "",
    };
    let mut extra_paths = Vec::new();
    if let Ok(runtime_lib) = env::var("G2P_TVM_RUNTIME_LIB") {
        let runtime_path = Path::new(&runtime_lib);
        if !runtime_path.exists() {
            panic!("G2P_TVM_RUNTIME_LIB not found: {runtime_lib}");
        }
        let dir = runtime_path
            .parent()
            .expect("G2P_TVM_RUNTIME_LIB must have a parent directory");
        extra_paths.push(dir.to_path_buf());
        let stem = runtime_path
            .file_stem()
            .expect("G2P_TVM_RUNTIME_LIB must have a file stem")
            .to_string_lossy();
        let lib_name = stem.strip_prefix("lib").unwrap_or(&stem);
        println!("cargo:rerun-if-changed={}", runtime_path.display());
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=dylib={}", lib_name);
    }

    if !os_env_var.is_empty() {
        let current = env::var(os_env_var).unwrap_or_default();
        let separator = if os_env_var == "PATH" { ";" } else { ":" };
        let mut paths = vec![lib_dir];
        for extra in extra_paths {
            paths.push(extra.to_string_lossy().to_string());
        }
        let extra_joined = paths.join(separator);
        let new_value = if current.is_empty() {
            extra_joined
        } else {
            format!("{current}{separator}{extra_joined}")
        };
        println!("cargo:rustc-env={}={}", os_env_var, new_value);
    }

    if let Ok(static_lib) = env::var("G2P_TVM_SYSTEM_LIB") {
        let static_path = Path::new(&static_lib);
        if !static_path.exists() {
            panic!("G2P_TVM_SYSTEM_LIB not found: {static_lib}");
        }
        let dir = static_path
            .parent()
            .expect("G2P_TVM_SYSTEM_LIB must have a parent directory");
        let stem = static_path
            .file_stem()
            .expect("G2P_TVM_SYSTEM_LIB must have a file stem")
            .to_string_lossy();
        let lib_name = stem.strip_prefix("lib").unwrap_or(&stem);
        println!("cargo:rerun-if-changed={}", static_path.display());
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static={}", lib_name);
        match env::var("CARGO_CFG_TARGET_OS").as_deref() {
            Ok("macos") => {
                println!(
                    "cargo:rustc-link-arg=-Wl,-force_load,{}",
                    static_path.display()
                );
            }
            Ok("linux") | Ok("android") => {
                println!("cargo:rustc-link-arg=-Wl,--whole-archive");
                println!("cargo:rustc-link-arg={}", static_path.display());
                println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
            }
            _ => {}
        }
    }
}
