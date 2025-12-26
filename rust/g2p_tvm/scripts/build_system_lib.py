#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.contrib import cc


def _ensure_prefix(prefix: str) -> str:
    if not prefix:
        raise ValueError("prefix must be non-empty")
    if not prefix.endswith("_"):
        return f"{prefix}_"
    return prefix


def build_system_lib(output_dir: Path, prefix: str) -> Path:
    prefix = _ensure_prefix(prefix)

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(x: R.Tensor((1,), "float32")) -> R.Tensor((1,), "float32"):
            return R.add(x, R.const(1.0, "float32"))

    mod = Mod.with_attr("system_lib_prefix", prefix)

    target = tvm.target.Target("llvm", host="llvm")
    exec_obj = relax.build(mod, target)

    output_dir.mkdir(parents=True, exist_ok=True)
    lib_name = f"lib{prefix.rstrip('_')}.a"
    lib_path = output_dir / lib_name
    exec_obj.export_library(str(lib_path), fcompile=cc.create_staticlib)
    return lib_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Relax system-lib static archive for testing.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rust/g2p_tvm/system_lib"),
        help="Directory to write the static library into.",
    )
    parser.add_argument(
        "--prefix",
        default="g2p_test_",
        help="System lib prefix to register (default: g2p_test_).",
    )
    args = parser.parse_args()

    lib_path = build_system_lib(args.output_dir, args.prefix)
    print(f"built={lib_path}")
    print(f"prefix={_ensure_prefix(args.prefix)}")


if __name__ == "__main__":
    main()
