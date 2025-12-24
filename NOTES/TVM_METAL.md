# TVM Metal Compile Notes

- `relax.build` defaults to the generic "default" pipeline, which does not apply GPU schedules.
- For `target="metal"`, the default GPU pipeline (`relax.get_default_pipeline(Target("metal"))`) adds dlight GPU schedules so TIR loops are bound to threads.
- Without the GPU pipeline, Metal builds can fail TIR memory verification with "Did you forget to bind?" on simple ops like reshape.
- The output extension is for the host runtime library; on macOS/iOS targets, `.dylib` is more idiomatic than `.so` even for Metal builds, since the device code is embedded in the host library.
- Target presets in `python/src/charsiug2p_tvm/config.py` now resolve default extensions (e.g., `metal-macos` -> `dylib`), while raw target strings still work.
- If Metal compilation fails inside `tvm/dlight/gpu/gemv.py` with "too many values to unpack", skip the DLight GEMV schedule via `--skip-dlight-gemv` (uses DLight fallback instead).
- If Metal compilation fails with `VerifyMemory` on scalar `cur_pos`/`T_add` (no thread binding), export KV-cache `cur_pos` as a 1-element tensor so DLight fallback can bind threads.
