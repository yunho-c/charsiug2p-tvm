use std::env;
use std::process::Command;

fn main() {
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
    if !os_env_var.is_empty() {
        let current = env::var(os_env_var).unwrap_or_default();
        let separator = if os_env_var == "PATH" { ";" } else { ":" };
        let new_value = if current.is_empty() {
            lib_dir.clone()
        } else {
            format!("{current}{separator}{lib_dir}")
        };
        println!("cargo:rustc-env={}={}", os_env_var, new_value);
    }
}
