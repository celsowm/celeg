# How to Add a Quantization Format

1. Define the source encoding in the checkpoint format layer.
2. Define the runtime encoding and its typed storage variant.
3. Add a materializer that maps source views to that runtime encoding.
4. Add decode and prefill kernels with explicit argument contracts.
5. Register dispatch by runtime encoding, not by checkpoint filename or model
   architecture.
6. Add numerical quality, memory, deterministic source-format, and performance
   tests before changing the default policy.
7. Report the selected runtime encoding in `ExecutionPlan` diagnostics.

A checkpoint parser must not know CUDA allocation details, and an operator must
not infer source-format semantics from a raw pointer.
