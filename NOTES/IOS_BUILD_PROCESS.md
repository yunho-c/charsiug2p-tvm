# iOS Build Integration: The Static Linking Flow

This document visualizes how the CharsiuG2P project integrates Rust, TVM, and Flutter for iOS builds. The key challenge is handling **static linking** (`.a` files) because iOS does not support dynamic loading of custom libraries easily, and TVM's `system-lib` feature requires manual symbol registration.

## Build Flow Diagram

```mermaid
flowchart TB
    subgraph Assets ["1. Artifact Staging (Manual/Script)"]
        direction TB
        ModelPy["Compile & Export Model\n(tvm_compile.py)"] --> G2pLib[("libg2p_system_lib.a\n(metal-ios)")]
        RuntimeBuild["Build TVM Runtime\n(CMake/iOS)"] --> RuntimeLib[("libtvm_runtime.a\n(ios-arm64)")]
        
        G2pLib --> AssetDir["flutter/assets/metal-ios/"]
        RuntimeLib --> AssetDir
    end

    subgraph RustBuild ["2. Rust Build (Cargo via builds_pod.sh)"]
        direction TB
        Env["Env Vars:\nTVM_STATIC_LINK=1\nTVM_FFI_STATIC=1\nTARGET=aarch64-apple-ios"]
        
        BuildRS["g2p_tvm/build.rs"]
        Env --> BuildRS
        
        TvmFFI["tvm-ffi crate\n(Rust Bindings)"]
        BuildRS -.->|"Links"| TvmFFI
        
        LibFFI[("libtvm_ffi.a")]
        TvmFFI -->|"Bundles"| LibFFI
        
        Code["Rust Code\n(g2p_pipeline, api.rs)"]
        LibFFI --> RustStatic[("libcharsiug2p_g2p_ffi.a")]
        Code --> RustStatic
        
        note1["NOTE: This contains ALL Rust code\nAND the TVM FFI C bindings wrapper.\nIt does NOT contain the TVM Runtime C++ code."]
        RustStatic --- note1
    end

    subgraph Xcode ["3. Xcode Linker (Podspec)"]
        direction TB
        Input1["libcharsiug2p_g2p_ffi.a\n(From Cargo)"]
        Input2["libtvm_runtime.a\n(From Assets)"]
        Input3["libg2p_system_lib.a\n(From Assets)"]
        
        Podspec["charsiug2p_flutter.podspec\nOTHER_LDFLAGS = -force_load ..."]
        
        Input1 --> Linker
        Input2 --> Linker
        Input3 --> Linker
        Podspec --> Linker
        
        Linker["ld (Linker)"]
        
        AppBinary[("Runner.app (iOS Binary)")]
        Linker --> AppBinary
        
        note2["-force_load is CRITICAL.\nIt ensures unused symbols (like __tvm_module_startup)\nare kept, registering the model in the global registry system."]
        Linker --- note2
    end

    Assets --> Xcode
    RustBuild --> Xcode
```

## Key Components

1.  **`libg2p_system_lib.a`**: The "Bytecode" of your model + generated C functions for operators. It registers itself via a global constructor (`__tvm_module_startup`).
2.  **`libtvm_runtime.a`**: The C++ engine that knows how to run the bytecode.
3.  **`libcharsiug2p_g2p_ffi.a`**: The bridge. It contains:
    *   The compiled Rust code.
    *   The `tvm-ffi` C bindings (tiny wrapper).
    *   It calls into `libtvm_runtime` functions (unresolved symbols at this stage).
    *   It tries to load modules from the "System Lib" registry (populated by component #1).

## The Linker's Job

The final link step merges all three.
*   Rust code calls `TVMFuncCall`.
*   Linker finds `TVMFuncCall` in `libtvm_runtime.a`.
*   Rust code calls `ExtensionNode::GetGlobal("ffi.SystemLib")`.
*   Runtime looks up "g2p_" prefix.
*   Because we used `-force_load` on `libg2p_system_lib.a`, its startup function ran and registered "g2p_..." in the runtime's internal map.
*   **Success!** The model loads as if it were a dynamic library.
