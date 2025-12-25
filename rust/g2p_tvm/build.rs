use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    let is_ios = target.contains("apple-ios");
    let static_requested = is_ios || env_flag("G2P_TVM_STATIC_LINK");

    println!("cargo:rerun-if-env-changed=G2P_TVM_SYSTEM_LIB");
    println!("cargo:rerun-if-env-changed=G2P_TVM_RUNTIME_LIB");
    println!("cargo:rerun-if-env-changed=G2P_TVM_FFI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=G2P_TVM_FFI_STATIC_NAME");
    println!("cargo:rerun-if-env-changed=G2P_TVM_LINK_FFI_TESTING");
    println!("cargo:rerun-if-env-changed=G2P_TVM_STATIC_LINK");

    let ffi_lib_dir = match env::var("G2P_TVM_FFI_LIB_DIR") {
        Ok(value) => PathBuf::from(value),
        Err(_) => {
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
            PathBuf::from(lib_dir)
        }
    };
    let lib_dir = ffi_lib_dir.to_string_lossy().to_string();
    if lib_dir.is_empty() {
        panic!("tvm-ffi-config returned an empty libdir");
    }
    println!("cargo:rustc-link-search=native={}", lib_dir);

    let link_testing = match env::var("G2P_TVM_LINK_FFI_TESTING") {
        Ok(value) => parse_bool(&value),
        Err(_) => !static_requested,
    };
    let ffi_static_name = env::var("G2P_TVM_FFI_STATIC_NAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            if is_ios {
                "tvm_ffi_static".to_string()
            } else {
                "tvm_ffi".to_string()
            }
        });
    if static_requested {
        link_static_lib(&ffi_static_name, &ffi_lib_dir);
        if link_testing {
            link_static_lib("tvm_ffi_testing", &ffi_lib_dir);
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=tvm_ffi");
        if link_testing {
            println!("cargo:rustc-link-lib=dylib=tvm_ffi_testing");
        }
    }

    let os_env_var = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "PATH",
        Ok("macos") => "DYLD_LIBRARY_PATH",
        Ok("linux") => "LD_LIBRARY_PATH",
        _ => "",
    };
    let mut extra_paths = Vec::new();
    if !static_requested {
        extra_paths.push(ffi_lib_dir.clone());
    }
    if let Ok(runtime_lib) = env::var("G2P_TVM_RUNTIME_LIB") {
        let runtime_path = Path::new(&runtime_lib);
        if !runtime_path.exists() {
            panic!("G2P_TVM_RUNTIME_LIB not found: {runtime_lib}");
        }
        let dir = runtime_path
            .parent()
            .expect("G2P_TVM_RUNTIME_LIB must have a parent directory");
        let (runtime_path, runtime_static) = resolve_runtime_lib(runtime_path, static_requested);
        let stem = runtime_path
            .file_stem()
            .expect("G2P_TVM_RUNTIME_LIB must have a file stem")
            .to_string_lossy();
        let lib_name = stem.strip_prefix("lib").unwrap_or(&stem);
        println!("cargo:rerun-if-changed={}", runtime_path.display());
        if runtime_static {
            link_static_lib(lib_name, dir);
        } else {
            println!("cargo:rustc-link-search=native={}", dir.display());
            extra_paths.push(dir.to_path_buf());
            println!("cargo:rustc-link-lib=dylib={}", lib_name);
        }
    }

    if !os_env_var.is_empty() {
        let current = env::var(os_env_var).unwrap_or_default();
        let separator = if os_env_var == "PATH" { ";" } else { ":" };
        let mut paths = Vec::new();
        for extra in extra_paths {
            paths.push(extra.to_string_lossy().to_string());
        }
        let extra_joined = paths.join(separator);
        if extra_joined.is_empty() {
            return;
        }
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
        link_static_lib(lib_name, dir);
    }
}

fn env_flag(name: &str) -> bool {
    env::var(name).map(|value| parse_bool(&value)).unwrap_or(false)
}

fn parse_bool(value: &str) -> bool {
    matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes")
}

fn is_static_lib(path: &Path) -> bool {
    matches!(path.extension().and_then(|ext| ext.to_str()), Some("a") | Some("lib"))
}

fn resolve_runtime_lib(path: &Path, static_requested: bool) -> (PathBuf, bool) {
    if is_static_lib(path) {
        return (path.to_path_buf(), true);
    }
    if !static_requested {
        return (path.to_path_buf(), false);
    }
    let dir = path
        .parent()
        .expect("G2P_TVM_RUNTIME_LIB must have a parent directory");
    let stem = path
        .file_stem()
        .expect("G2P_TVM_RUNTIME_LIB must have a file stem")
        .to_string_lossy();
    let lib_name = stem.strip_prefix("lib").unwrap_or(&stem);
    let candidate = dir.join(format!("lib{lib_name}.a"));
    if candidate.exists() {
        return (candidate, true);
    }
    panic!(
        "Static linking requested but no static lib found for {} in {}",
        lib_name,
        dir.display()
    );
}

fn link_static_lib(lib_name: &str, dir: &Path) {
    println!("cargo:rustc-link-search=native={}", dir.display());
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => {
            println!("cargo:rustc-link-lib=static={}", lib_name);
        }
        _ => {
            println!("cargo:rustc-link-lib=static:+whole-archive={}", lib_name);
        }
    }
}
