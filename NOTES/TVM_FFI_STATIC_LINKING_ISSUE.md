---
name: "🐛 Bug report"
about: Please include a description of your environment, preferably a minimum script to reproduce the problem. Find the list of label tags at https://tinyurl.com/5fnd5rkn.
title: "[Bug] tvm-ffi-sys build.rs forces dynamic libtvm_ffi and blocks static linking"
labels: "needs-triage, type: bug"
---

Thanks for participating in the TVM community! We use https://discuss.tvm.ai for any general usage questions and discussions. The issue tracker is used for actionable items such as feature proposals discussion, roadmaps, and bug tracking.  You are always welcomed to post on the forum first :smile_cat:

Issues that are inactive for a period of time may get closed. We adopt this policy so that we won't lose track of actionable issues that may fall at the bottom of the pile. Feel free to reopen a new one if you feel there is an additional problem that needs attention when an old one gets closed.

### Expected behavior

When a static `libtvm_ffi_static.a` is available, the Rust crate `tvm-ffi-sys` should be able to link it instead of `libtvm_ffi.dylib/.so`. Ideally this is controlled by canonical env flags, e.g.:

- `TVM_FFI_STATIC=1` (prefer static linking)
- `TVM_FFI_LIB_DIR=/path/to/tvm-ffi/build/lib` (override `tvm-ffi-config --libdir`)
- `TVM_FFI_LINK_TESTING=1` (opt-in to `tvm_ffi_testing` when needed)

This is required for iOS/App Store builds where dynamic loading of non-system dylibs is not allowed.

### Actual behavior

`tvm-ffi-sys` always emits `cargo:rustc-link-lib=dylib=tvm_ffi` (and `tvm_ffi_testing`) in `rust/tvm-ffi-sys/build.rs`, so the resulting crate still depends on the dynamic library even if a static archive is present.

### Environment

- OS: macOS 14.x (also relevant for iOS targets)
- TVM/tvm-ffi: current `apache/tvm-ffi` (tested on commit `ae346ec`)
- Rust: 1.78+
- Target: `aarch64-apple-ios` / `aarch64-apple-darwin`

### Steps to reproduce

1. Build tvm-ffi with static archive output (default provides `libtvm_ffi_static.a`).
2. Run:

   ```bash
   TVM_FFI_LIB_DIR=/path/to/tvm-ffi/build/lib \
   cargo build -p tvm-ffi-sys
   ```

3. Observe that `build.rs` still links `dylib=tvm_ffi` and `dylib=tvm_ffi_testing`.

### Triage

* needs-triage
* type: bug
