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

