# TVM Profiling Notes

- Profiling now records encoder vs decoder time separately using a timed G2P path in `tvm_runtime`.
- CSV output includes encoder/decoder totals and per-sample milliseconds alongside the existing aggregate metrics.
