# charsiug2p-tvm (Python)

Modern Python 3.12 project scaffold using:

- `pyproject.toml` (PEP 621) for packaging metadata
- `pixi` (manifest stored in `pyproject.toml`, no `pixi.toml`)
- `hatch` / `hatchling` for builds and dev workflows
- `ruff` for linting + formatting (`line-length = 120`)

## Quickstart (pixi)

```bash
cd python
pixi install
pixi run test
pixi run lint
pixi run fmt
```

## Quickstart (hatch)

```bash
cd python
hatch run test
hatch run lint
hatch run fmt
```

## Running

```bash
python -m charsiug2p_tvm --version
charsiug2p-tvm --version
```

## Runnable commands (pixi)

```bash
pixi run python -m charsiug2p_tvm info
pixi run python -m charsiug2p_tvm run --lang eng-us Char siu
pixi run python -m charsiug2p_tvm compile --target llvm --output-ext so
# compile multiple batch sizes (comma-separated or repeatable)
pixi run python -m charsiug2p_tvm compile --target llvm --batch-sizes 1,4,16
# run inference using compiled TVM artifacts
pixi run python -m charsiug2p_tvm run-tvm --lang eng-us Char siu
# run inference using multiple compiled batch sizes
pixi run python -m charsiug2p_tvm run-tvm --lang eng-us Char siu --batch-sizes 1,4,16
# compare TVM outputs against reference on a TSV or directory
pixi run python -m charsiug2p_tvm verify ../external/CharsiuG2P/data/test --limit 100 --tvm-target llvm
# verify with multiple compiled batch sizes
pixi run python -m charsiug2p_tvm verify ../external/CharsiuG2P/data/test --limit 100 --tvm-target llvm --batch-sizes 1,4,16
# profile multiple targets and save results
pixi run python -m charsiug2p_tvm profile ../external/CharsiuG2P/data/test --target llvm --runs 3 --limit 100
# comma-separated targets are also supported
pixi run python -m charsiug2p_tvm profile ../external/CharsiuG2P/data/test --target llvm,cuda --runs 3 --limit 100
# profile using multiple compiled batch sizes
pixi run python -m charsiug2p_tvm profile ../external/CharsiuG2P/data/test --target llvm --runs 3 --limit 100 --batch-sizes 1,4,16
# profile using KV-cache artifacts
pixi run python -m charsiug2p_tvm profile ../external/CharsiuG2P/data/test --target llvm --runs 3 --limit 100 --kv-cache
# or, explicitly specify the output directory
pixi run python -m charsiug2p_tvm compile --output-dir dist/tvm/byt5_tiny_8 --target llvm --output-ext so
```
