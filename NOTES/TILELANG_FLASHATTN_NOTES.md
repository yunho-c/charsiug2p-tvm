# TileLang FlashAttention Notes

Source: TileLang docs/examples (Context7: /tile-ai/tilelang).

## Key FlashAttention structure (TileLang)

- Uses `@T.prim_func` + `T.Kernel` to define a tiled, GPU-kernel-like program with a 3D grid `(bx, by, bz)`:
  - `bx`: block index over sequence length (M)
  - `by`: head dimension
  - `bz`: batch dimension
- Allocates staged buffers:
  - `T.alloc_shared` for Q/K/V tiles
  - `T.alloc_fragment` for accumulator and temporary registers (scores, logsum, partial output)
- Loads Q tile once per block and loops over K/V tiles with `T.Pipelined(..., num_stages=...)` to overlap memory loads and compute.

## Core FlashAttention math (blockwise, numerically stable)

- Compute `acc_s = Q @ K^T` per block.
- Maintain running `scores_max`, `scores_scale`, and `logsum` to keep softmax stable across blocks.
- Rescale previous partial output before accumulating new `acc_s @ V`.
- Final output normalization: `acc_o /= logsum`.

## TileLang primitives used in FlashAttention examples

- `T.copy` for explicit loads/stores between global, shared, and registers.
- `T.gemm` for block GEMM (supports `transpose_B=True`).
- `T.reduce_max`, `T.reduce_sum` for softmax statistics.
- `T.exp2` for fast exponentiation in softmax.
- `T.clear`, `T.fill` to reset accumulators.
- `T.if_then_else` for causal masking (upper triangle -> `-inf`).
- `T.Parallel` and `T.serial` for parallel vs serial loops.

## Memory/layout optimizations

- `T.annotate_layout({buf: tl.layout.make_swizzled_layout(buf)})` to reduce shared-memory bank conflicts.
- `T.use_swizzle(panel_size=..., order=...)` to improve L2 locality via threadblock swizzling.
- Examples suggest moving some buffers from shared memory to registers on memory‑constrained GPUs (e.g., MI300X).

## Other relevant TileLang features

- GEMM kernels are defined similarly with shared + fragment buffers and `T.Pipelined` loops.
- Analysis tools (`tilelang.tools.Analyzer`) can estimate FLOPs and bandwidth from TVM IR.
- TileLang can use prebuilt TVM libraries or build TVM from source. Environment variables: `PYTHONPATH`, `TVM_IMPORT_PYTHON_PATH`.

## Implications for this project

- TileLang FlashAttention kernels are explicit, block‑tiled, and rely on manual buffer/control flow.
- Adopting FlashAttention for G2P decoder would likely mean introducing a custom kernel (instead of relying on Relax + default schedules).
- This is a later optimization step; baseline correctness via Relax is still the priority.
