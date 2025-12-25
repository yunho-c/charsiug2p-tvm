Here is a complete, step-by-step tutorial summarizing our conversation.

This guide focuses on the **"Single Binary" Strategy**. We will compile multiple TVM (Relax) models into static libraries, merge them, and link them directly into your Rust crate. This ensures full compatibility with iOS (which forbids JIT/dynamic loading) and simplifies Android deployment.

---

# Tutorial: integrating Multiple TVM (Relax) Models into Flutter via Rust

**Goal:** Run multiple unique AI models (e.g., YOLO + BERT) on iOS and Android within a single Flutter app.
**Tools:** Apache TVM (Relax), Rust (with `flutter_rust_bridge`), Flutter.
**Architecture:**

1. **Python:** Rename model entry points  Pack into one Static Library (`.a`).
2. **Rust:** Link the library  Load models via `SystemLibModule`.
3. **App:** Flutter calls Rust; Rust runs the models.

---

## Phase 1: The Build Script (Python)

*Running on your development machine.*

You cannot ship raw models to iOS. You must compile them ahead-of-time (AOT). Crucially, you must **rename the entry function** of each model so they don't fight for the name `main` when linked together.

**`build_models.py`**

```python
import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.contrib import cc

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
# Define your target. 
# CRITICAL: "system-lib" must be enabled. 
# This tells TVM: "Register functions globally; don't wait for dlopen."
target_ios = tvm.target.Target("llvm -mtriple=arm64-apple-ios -system-lib")
# For Android, use a target like: "llvm -mtriple=aarch64-linux-android -system-lib"

# ---------------------------------------------------------
# 2. Helper: Rename Entry Point
# ---------------------------------------------------------
def prepare_model(mod, prefix):
    """
    Renames the main function (e.g., 'main') to '{prefix}_main'
    to prevent symbol collision in the final binary.
    """
    new_name = f"{prefix}_main"
    
    # 1. Point the new name to the existing function
    mod[new_name] = mod["main"]
    
    # 2. Delete the old default entry point
    del mod["main"]
    
    return mod, new_name

# ---------------------------------------------------------
# 3. Compile & Merge
# ---------------------------------------------------------
# Assume you have two Relax modules: mod_yolo and mod_bert

# A. Prepare them
mod_yolo, entry_yolo = prepare_model(mod_yolo, "yolo")
mod_bert, entry_bert = prepare_model(mod_bert, "bert")

# B. Build them
# Note: You can build them into a single module using import_module
# or keep them separate. Merging them here is cleanest.
mod_yolo.import_module(mod_bert) 

# C. Compile the PARENT module (which now contains the child)
ex = relax.build(mod_yolo, target=target_ios)

# ---------------------------------------------------------
# 4. Export to Static Library (.a)
# ---------------------------------------------------------
output_path = "../rust/models/libcombined_models.a"

ex.export_library(
    output_path,
    fcompile=cc.create_staticlib,
    # Optional: cross-compiler flags if needed
    # options=["-arch", "arm64", "-isysroot", "/path/to/iPhoneOS.sdk"] 
)

print(f"Exported to {output_path}")
print(f"Entry Points: {entry_yolo}, {entry_bert}")

```

---

## Phase 2: Link in Rust

*Configuring your `flutter_rust_bridge` project.*

Now you have `libcombined_models.a`. You need to tell the Rust compiler to bake this file into your app.

**`rust/build.rs`**

```rust
fn main() {
    // 1. Where to find the library?
    // Ensure this path is correct relative to your Cargo.toml
    println!("cargo:rustc-link-search=native=models/");

    // 2. Link the library (libcombined_models.a -> "combined_models")
    println!("cargo:rustc-link-lib=static=combined_models");
    
    // 3. (Optional) Force C++ linking if TVM relies on stdlibc++
    // println!("cargo:rustc-link-lib=c++"); // macOS/iOS
    // println!("cargo:rustc-link-lib=stdc++"); // Android/Linux
}

```

---

## Phase 3: Load in Rust

*Writing the runtime logic.*

Because we used `system-lib`, the models are already in memory when the app starts. We access them using the global `SystemLibModule` registry and the unique names we defined in Phase 1.

**`rust/src/api.rs` (or your preferred file)**

```rust
use tvm::runtime::SystemLibModule;
use tvm::runtime::Module; // Standard module trait
use flutter_rust_bridge::frb;

// This function is callable from Dart
#[frb]
pub fn run_inference(model_type: String) -> anyhow::Result<String> {
    // 1. Access the Global Registry
    let sys_lib = SystemLibModule::default();

    // 2. Determine which entry point to load
    let entry_func_name = match model_type.as_str() {
        "yolo" => "yolo_main",
        "bert" => "bert_main",
        _ => return Err(anyhow::anyhow!("Unknown model type")),
    };

    // 3. Get the VM Executable
    let vm_exec = sys_lib.get_function(entry_func_name)
        .expect("Entry point not found! Did you link the static lib?");

    // 4. Initialize Relax VM
    let vm_factory = Module::load("relax.VirtualMachine")?;
    let device = tvm::runtime::Device::cpu(0);
    
    // "init" takes [executable, device]
    let mut vm = vm_factory
        .get_function("init")?
        .invoke(vec![vm_exec, device.into()])?;

    // 5. Run (Pseudo-code)
    // let input = ...;
    // let output = vm.get_function("main")?.invoke(vec![input])?;
    
    Ok("Inference complete".to_string())
}

```

---

## Phase 4: Flutter Integration

*The easy part.*

Since you used **Static Linking** in Rust, standard Flutter build commands work out of the box.

1. **iOS:**
* `flutter_rust_bridge` automatically links your Rust crate (which now contains the TVM code) into the Runner.
* No Drag-and-Drop in Xcode required.
* No custom `.podspec` required.


2. **Android:**
* Rust compiles to `librust_lib.so` (Dynamic Lib).
* Because we linked `libcombined_models.a` into that Rust `.so`, the models are hidden inside it.
* Deploy as usual.



**Dart Code (`lib/main.dart`):**

```dart
import 'ffi.dart'; // generated by FRB

void main() async {
  // Just call the Rust function
  final result = await api.runInference(modelType: "yolo");
  print(result);
}

```

### Troubleshooting Checklist

* **"Symbol not found" (Linker Error):**
* Did you set `cargo:rustc-link-search` correctly in `build.rs`?
* Did you use `system-lib=True` in Python?


* **"Function not found" (Runtime Error):**
* Did you rename the entry point in Python?
* Are you requesting the *exact* string (`yolo_main`) in Rust that you set in Python?


* **iOS Build Failures:**
* Ensure the `.a` file was compiled for the correct architecture (`arm64` for device, `x86_64` or `arm64` for simulator). You may need to create a "Fat" library (using `lipo`) if you want one file to work on both Simulator and Device during dev.

---

## Validation notes (macOS prototype)

I validated the system-lib flow against TVM source/tests and a local Rust prototype. A few corrections to the tutorial above:

1) You do not need to rename `main` if you use `system_lib_prefix`. The prefix is how TVM keeps multiple system-lib modules from colliding, and the VM entry remains `main`.
2) In Rust (tvm-ffi), there is no `SystemLibModule::default()` helper. Instead, call the global function `ffi.SystemLib` and pass the prefix, then treat the returned `Module` like any other VM executable.
3) There is no `Module::load("relax.VirtualMachine")` path. For Relax VM, you must call `vm_load_executable` + `vm_initialization` on the module, then fetch the entry function. The existing `TvmModule::load_entry` logic already does this.
4) Static libraries must be force-loaded by the linker, otherwise their registration code is dropped. On macOS use `-Wl,-force_load,<path/to/lib.a>`. On ELF targets use `-Wl,--whole-archive`/`--no-whole-archive`.
5) The Relax VM loader is registered by `libtvm_runtime` (or `libtvm`). Linking only `libtvm_ffi` is not enough; you must link a TVM runtime library that contains `ffi.Module.load_from_bytes.relax.VMExecutable`.

### macOS prototype steps used

Build a tiny Relax VM system-lib:

```bash
py -3.12 rust/g2p_tvm/scripts/build_system_lib.py --output-dir rust/g2p_tvm/system_lib --prefix g2p_test_
```

Link it into the Rust crate and run the smoke test:

```bash
DYLD_LIBRARY_PATH=/path/to/tvm/build:/path/to/tvm-ffi/build/lib:$DYLD_LIBRARY_PATH \
G2P_TVM_RUNTIME_LIB=/path/to/tvm/build/libtvm_runtime.dylib \
G2P_TVM_SYSTEM_LIB=rust/g2p_tvm/system_lib/libg2p_test.a \
G2P_TVM_SYSTEM_LIB_PREFIX=g2p_test_ \
cargo test -p charsiug2p-g2p-tvm system_lib_smoke
```

The test loads the system lib, initializes the VM, calls `main`, and checks that `1.5 -> 2.5` as expected.

### System-lib compile path (Python)

The Python `compile` command now supports a system-lib export that emits:

- `libg2p_system_lib.a` (static archive containing encoder/decoder/prefill/step)
- `system_lib_metadata.json` (prefix mapping for Rust)

Example (macOS, cacheless):

```bash
py -3.12 -m charsiug2p_tvm compile \
  --checkpoint charsiu/g2p_multilingual_byT5_tiny_8_layers_100 \
  --target llvm \
  --system-lib \
  --system-lib-prefix g2p_
```

The metadata JSON contains the prefixes that Rust uses via `ArtifactResolver::resolve_system_lib_prefixes()`.
The current implementation uses `libtool` for combining static archives, so the one-shot `--system-lib` export is macOS-only for now.

### Additional findings

- The `ffi.SystemLib` entrypoint is registered by tvm-ffi; it returns a `Module` that resolves symbols by prefix (with and without `__tvm_ffi_`).
- Relax VM deserialization (`ffi.Module.load_from_bytes.relax.VMExecutable`) is registered in `libtvm_runtime` (or `libtvm`), so tvm-ffi alone is insufficient for system-lib execution.
- Force-loading the static archive is required to keep `__tvm_module_startup` and registration symbols from being stripped; without it, the runtime fails with `vm_load_executable` missing from `<system-lib:...>`.
- Using `cargo:rustc-link-lib=static:+whole-archive=...` in `rust/g2p_tvm/build.rs` is the reliable way to force-load across dependent crates (the previous `rustc-link-arg=-Wl,-force_load,...` did not propagate to `g2p_cli`).
- The Rust build now supports `G2P_TVM_RUNTIME_LIB` to link `libtvm_runtime.dylib` and `G2P_TVM_SYSTEM_LIB` to link a static system-lib archive (plus whole-archive semantics).
- Static linking for the TVM runtime/ffi can be enabled with `G2P_TVM_STATIC_LINK=1`; optionally set `G2P_TVM_FFI_LIB_DIR` (and `G2P_TVM_FFI_STATIC_NAME` on iOS) to point at static tvm-ffi builds. `G2P_TVM_LINK_FFI_TESTING=1` re-enables linking tvm-ffi testing symbols when needed.

### `system_lib_metadata.json` schema

```json
{
  "archive": "libg2p_system_lib.a",
  "prefix_base": "g2p_",
  "prefixes": {
    "encoder": "g2p_encoder_",
    "decoder": "g2p_decoder_"
  },
  "checkpoint": "charsiu/g2p_multilingual_byT5_tiny_8_layers_100",
  "batch_size": 1,
  "max_input_bytes": 64,
  "max_output_len": 128,
  "target": "llvm",
  "output_ext": "so",
  "use_kv_cache": false
}
```

When KV-cache is enabled, `decoder` is omitted and `decoder_prefill`/`decoder_step` are present in `prefixes`.

Rust CLI can now opt into system-lib loading (single batch size only):

```bash
G2P_TVM_SYSTEM_LIB=/path/to/libg2p_system_lib.a \
G2P_TVM_RUNTIME_LIB=/path/to/libtvm_runtime.dylib \
cargo run -p charsiug2p-g2p-cli -- --system-lib --tvm-target llvm --lang eng-us Char siu
```
