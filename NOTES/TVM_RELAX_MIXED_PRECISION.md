# TVM Relax Mixed Precision Notes

- Use `relax.transform.ToMixedPrecision` when working in Relax.
- The pass assumes the module starts in fp32 and mainly operates inside dataflow blocks.
- Typical sequence: `ConvertToDataflow` -> `ToMixedPrecision(out_dtype=...)` -> `ToNonDataflow`.
- `fp16_input_names` lets you change specific function parameters to fp16 in the signature.
