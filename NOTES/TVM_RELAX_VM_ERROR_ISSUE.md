# Draft GitHub Issue: Relax VM executable loader error text

Title: [Bug] Improve error message when loading Relax VM executables without VM runtime

## Expected behavior

When loading a Relax VM executable (e.g., a .so built from Relax) without the VM loader registered, the error should clearly explain that the artifact is a Relax VM executable and must be invoked through the Relax VM initialization flow (e.g., vm_load_executable + vm_initialization) or that libtvm_runtime/libtvm must be loaded to register the VM loader.

## Actual behavior

The error is cryptic for users integrating via FFI:

```
RuntimeError: Library binary was created using {relax.VMExecutable} but a loader of that name is not registered. Make sure to have runtime that registers ffi.Module.load_from_bytes.relax.VMExecutable
```

This does not clearly indicate that the functions are not in the library module and that users must call the VM loader initialization sequence (or use relax.VirtualMachine in Python).

## Environment

- OS: Ubuntu 22.04 (x86_64)
- TVM: Unity-era TVM built from source (external/tvm), built with LLVM enabled
- Runtime libs: libtvm_runtime.so + libtvm_ffi (tvm-ffi 0.1.0-alpha.0)
- Language: Rust via tvm-ffi crate (0.1.0-alpha.0), also reproducible from Python with tvm.runtime.load_module

## Steps to reproduce

1) Compile a Relax model to a VM executable (e.g., encoder/decoder) using Relax tooling (tvm.relax.frontend + tvm.compile).
2) Attempt to load the produced .so via a direct module loader (e.g., tvm.runtime.load_module("encoder.so") or tvm_ffi::Module::load_from_file("encoder.so") from Rust).
3) Observe the error above.

## Triage

* needs-triage
* type: bug

Additional context: this error arises because the module is a Relax VM executable and the entry function is not in the library module. The correct flow is to call vm_load_executable to get the VM module, then vm_initialization, and finally fetch the entry function from the VM module. A clearer error message could explicitly mention this.
