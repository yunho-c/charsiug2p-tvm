#!/bin/bash
# Build TVM native libraries from source
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

if [[ -n "${TVM_DIR:-}" ]]; then
    if [[ ! -d "$TVM_DIR" ]]; then
        echo "ERROR: TVM_DIR does not exist: $TVM_DIR" >&2
        exit 1
    fi
    TVM_DIR="$(cd "$TVM_DIR" && pwd)"
else
    TVM_DIR="${REPO_ROOT}/external/tvm"
fi

if [[ ! -d "$TVM_DIR" ]]; then
    echo "ERROR: TVM source directory not found: $TVM_DIR" >&2
    echo "Set TVM_DIR=/path/to/tvm if your checkout is elsewhere." >&2
    exit 1
fi

BUILD_DIR="${TVM_DIR}/build"

echo "=== Building TVM ==="
echo "TVM source: ${TVM_DIR}"
echo "Build directory: ${BUILD_DIR}"

# Initialize git submodules (required for tvm-ffi and other dependencies)
echo "Initializing git submodules..."
if git -C "$TVM_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$TVM_DIR" submodule update --init --recursive
else
    echo "Skipping submodule init: TVM checkout is not a git worktree."
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_LLVM=ON \
    -DUSE_CUDA=OFF \
    -DUSE_METAL=OFF \
    -DUSE_VULKAN=OFF \
    -DUSE_OPENCL=OFF

# Build using all available cores
cmake --build . --parallel "$(getconf _NPROCESSORS_ONLN)"

echo "=== TVM build complete ==="
echo "If using pixi, run 'pixi run build-tvm' from the python/ directory to rebuild."
