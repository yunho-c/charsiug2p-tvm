# TVM FFI (Rust) Notes

These notes capture what we learned from TVM-FFI (Context7) and the vendored Rust crates under `external/tvm/3rdparty/tvm-ffi/rust/`.

## Key takeaways

- TVM-FFI provides a stable C ABI with Rust bindings (`tvm-ffi` / `tvm-ffi-sys`) and a `Module` API that can load dynamic libraries and call packed functions.
- The Rust workspace is experimental and expects `tvm-ffi-config --libdir` to find `libtvm_ffi`, then links against `tvm_ffi` and `tvm_ffi_testing`.
- The Rust API includes `Tensor`, `Function`, and `Module` helpers, plus `Tensor::from_slice` for CPU allocations.
- Relax VM entrypoints appear to live in `libtvm_runtime` / `libtvm`, not `libtvm_ffi`, and the tvm-ffi global registry does not automatically see `relax.VirtualMachine`.

## Implications for this project

- For Relax bytecode (`relax.build(..., exec_mode="bytecode")`), we likely need `libtvm_runtime` (or `libtvm`) to construct the VM and invoke functions; `libtvm_ffi` alone may not expose the VM entrypoints.
- The tvm-ffi registry appears separate from the TVM runtime registry, so loading `libtvm_runtime` dynamically does not automatically surface `relax.VirtualMachine` through tvm-ffi.
- Mobile packaging likely needs to ship `libtvm_runtime` alongside `libtvm_ffi`, or use the TVM runtime C API directly for VM creation and invocation.
