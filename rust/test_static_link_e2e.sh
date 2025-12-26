#!/bin/bash
set -e

# ==============================================================================
# verify_system_lib.sh
#
# A script to verify the End-to-End static linking of the CharsiuG2P model.
# Defaults to 'llvm' (CPU) but supports other targets via env vars.
#
# Usage:
#   ./verify_system_lib.sh
#   TVM_TARGET=metal-macos TVM_DEVICE=metal ./verify_system_lib.sh
#
# Prerequisites:
#   - python (with charsiug2p_tvm installed)
#   - rust (cargo)
#   - Pre-built TVM and TVM-FFI static libraries
# ==============================================================================

REPO_ROOT=$(git rev-parse --show-toplevel)
RUST_ROOT="$REPO_ROOT/rust"

# Configuration
TVM_TARGET="${TVM_TARGET:-llvm}"
TVM_DEVICE="${TVM_DEVICE:-cpu}"
echo "Config: TARGET=$TVM_TARGET, DEVICE=$TVM_DEVICE"

# 1. Locate TVM Artifacts
if [ -z "$TVM_ROOT_DIR" ]; then
    POSSIBLE_TVM_ROOTS=(
        "/Users/yunhocho/GitHub/kokoro-tvm/reference/tvm"
        "$REPO_ROOT/external/tvm"
    )
    for p in "${POSSIBLE_TVM_ROOTS[@]}"; do
        if [ -d "$p/build" ]; then
            TVM_ROOT_DIR="$p"
            break
        fi
    done
fi

if [ -z "$TVM_ROOT_DIR" ]; then
    echo "Error: Could not locate TVM build directory."
    echo "Please set TVM_ROOT_DIR to the path containing 'build/libtvm_runtime.a'."
    exit 1
fi

TVM_RUNTIME_LIB="$TVM_ROOT_DIR/build/libtvm_runtime.a"
TVM_FFI_LIB_DIR="$TVM_ROOT_DIR/3rdparty/tvm-ffi/build/lib"

if [ ! -f "$TVM_RUNTIME_LIB" ]; then
    echo "Error: libtvm_runtime.a not found at $TVM_RUNTIME_LIB"
    exit 1
fi
if [ ! -d "$TVM_FFI_LIB_DIR" ]; then
    echo "Error: TVM FFI lib dir not found at $TVM_FFI_LIB_DIR"
    exit 1
fi

echo "Using TVM Root: $TVM_ROOT_DIR"

# 2. Check for Model Artifact
#    Adjust path based on target
MODEL_DIR="$REPO_ROOT/dist/tvm/charsiu_g2p_multilingual_byT5_tiny_8_layers_100/b1_in64_out128/$TVM_TARGET"
LIB_A="$MODEL_DIR/libg2p_system_lib.a"

if [ ! -f "$LIB_A" ]; then
    echo "Model artifact not found at: $LIB_A"
    echo "Attempting to compile..."
    cd "$REPO_ROOT/python"
    
    if command -v py >/dev/null; then
        PY_CMD="py -3.12"
    else
        PY_CMD="python3"
    fi
    
    $PY_CMD -m charsiug2p_tvm compile \
        --target "$TVM_TARGET" \
        --system-lib \
        --batch-size 1 \
        --max-input-bytes 64 \
        --max-output-len 128
        
    if [ ! -f "$LIB_A" ]; then
        echo "Error: Compilation failed to produce $LIB_A"
        exit 1
    fi
fi

# 3. Setup System Lib Metadata
#    We use the default convention "g2p_"
PREFIX="g2p_"
echo "Using System Lib Prefix: $PREFIX"

# 4. Run Rust Verification
echo "---------------------------------------------------"
echo "Running g2p_cli with static linking..."
echo "---------------------------------------------------"

export TVM_STATIC_LINK=1
export TVM_FFI_STATIC=1
export TVM_FFI_LIB_DIR="$TVM_FFI_LIB_DIR"
export TVM_RUNTIME_LIB="$TVM_RUNTIME_LIB"
export TVM_SYSTEM_LIB="$LIB_A"
export TVM_SYSTEM_LIB_PREFIX="$PREFIX"

cd "$RUST_ROOT"
# Note: cargo run re-compiles the binary with the new env vars provided
cargo run -p charsiug2p-g2p-cli --quiet -- \
  --system-lib \
  --tvm-target "$TVM_TARGET" \
  --device "$TVM_DEVICE" \
  --lang eng-us \
  "Char" "siu"

echo ""
echo "---------------------------------------------------"
echo "Verification Complete."
