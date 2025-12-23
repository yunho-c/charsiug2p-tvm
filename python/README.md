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
# run inference using compiled TVM artifacts
pixi run python -m charsiug2p_tvm run-tvm --lang eng-us Char siu
# or, explicitly specify the output directory
pixi run python -m charsiug2p_tvm compile --output-dir dist/tvm/byt5_tiny_8 --target llvm --output-ext so
```
