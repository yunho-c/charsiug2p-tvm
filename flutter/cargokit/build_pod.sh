#!/bin/sh
set -e

BASEDIR=$(dirname "$0")

# Workaround for https://github.com/dart-lang/pub/issues/4010
BASEDIR=$(cd "$BASEDIR" ; pwd -P)

# Remove XCode SDK from path. Otherwise this breaks tool compilation when building iOS project
NEW_PATH=`echo $PATH | tr ":" "\n" | grep -v "Contents/Developer/" | tr "\n" ":"`

export PATH=${NEW_PATH%?} # remove trailing :

env

# Platform name (macosx, iphoneos, iphonesimulator)
export CARGOKIT_DARWIN_PLATFORM_NAME=$PLATFORM_NAME

# Arctive architectures (arm64, armv7, x86_64), space separated.
export CARGOKIT_DARWIN_ARCHS=$ARCHS

# Current build configuration (Debug, Release)
export CARGOKIT_CONFIGURATION=$CONFIGURATION

# Path to directory containing Cargo.toml.
export CARGOKIT_MANIFEST_DIR=$PODS_TARGET_SRCROOT/$1

# Temporary directory for build artifacts.
export CARGOKIT_TARGET_TEMP_DIR=$TARGET_TEMP_DIR

# Output directory for final artifacts.
export CARGOKIT_OUTPUT_DIR=$PODS_CONFIGURATION_BUILD_DIR/$PRODUCT_NAME

# Directory to store built tool artifacts.
export CARGOKIT_TOOL_TEMP_DIR=$TARGET_TEMP_DIR/build_tool

# Directory inside root project. Not necessarily the top level directory of root project.
export CARGOKIT_ROOT_PROJECT_DIR=$SRCROOT

FLUTTER_EXPORT_BUILD_ENVIRONMENT=(
  "$PODS_ROOT/../Flutter/ephemeral/flutter_export_environment.sh" # macOS
  "$PODS_ROOT/../Flutter/flutter_export_environment.sh" # iOS
)

for path in "${FLUTTER_EXPORT_BUILD_ENVIRONMENT[@]}"
do
  if [[ -f "$path" ]]; then
    source "$path"
  fi
done

if [ "$CARGOKIT_DARWIN_PLATFORM_NAME" = "iphoneos" ] || [ "$CARGOKIT_DARWIN_PLATFORM_NAME" = "iphonesimulator" ]; then
  export TVM_STATIC_LINK=1
  export TVM_FFI_STATIC=1
  # TVM runtime and FFI are provided by tvm_runtime_flutter pod
fi

sh "$BASEDIR/run_build_tool.sh" build-pod "$@"

# macOS: stage TVM FFI dylibs so the Flutter app can link/load them via rpath.
if [ "$CARGOKIT_DARWIN_PLATFORM_NAME" = "macosx" ]; then
  # Resolve the TVM FFI build output from tvm-ffi-config.
  if ! command -v tvm-ffi-config >/dev/null 2>&1; then
    echo "error: tvm-ffi-config not found in PATH" >&2
    exit 1
  fi
  TVM_FFI_LIBDIR="$(tvm-ffi-config --libdir)"
  TVM_FFI_DYLIB="${TVM_FFI_LIBDIR}/libtvm_ffi.dylib"
  TVM_FFI_TESTING_DYLIB="${TVM_FFI_LIBDIR}/libtvm_ffi_testing.dylib"
  # Copy dylibs into the build products (and fail early if missing).
  for lib in "$TVM_FFI_DYLIB" "$TVM_FFI_TESTING_DYLIB"; do
    if [ ! -f "$lib" ]; then
      echo "error: missing dylib at ${lib}" >&2
      exit 1
    fi
    cp "$lib" "${BUILT_PRODUCTS_DIR}/"
  done
  # Also embed them under Frameworks and make the install name rpath-relative.
  if [ -n "$FRAMEWORKS_FOLDER_PATH" ]; then
    FRAMEWORKS_DIR="${BUILT_PRODUCTS_DIR}/${FRAMEWORKS_FOLDER_PATH}"
    mkdir -p "$FRAMEWORKS_DIR"
    for lib in "$TVM_FFI_DYLIB" "$TVM_FFI_TESTING_DYLIB"; do
      dest="${FRAMEWORKS_DIR}/$(basename "$lib")"
      cp "$lib" "$dest"
      install_name_tool -id "@rpath/$(basename "$lib")" "$dest"
    done
  fi
fi

# Make a symlink from built framework to phony file, which will be used as input to
# build script. This should force rebuild (podspec currently doesn't support alwaysOutOfDate
# attribute on custom build phase)
ln -fs "$OBJROOT/XCBuildData/build.db" "${BUILT_PRODUCTS_DIR}/cargokit_phony"
ln -fs "${BUILT_PRODUCTS_DIR}/${EXECUTABLE_PATH}" "${BUILT_PRODUCTS_DIR}/cargokit_phony_out"
