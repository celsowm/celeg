#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
void run_attention_tests(celeg::CudaStream& stream);
/// Exercises attention data-movement helpers (packed gate extract, QKV split).
void run_attention_data_movement_tests(celeg::CudaStream& stream);
/// Exercises the packed Q+Gate decode op sequence the CUDA decode path must
/// honor: per-head gate extraction, QK-norm/RoPE prepare, contiguous decode,
/// and sigmoid gate apply -- in that order. Feeding the packed buffer (or a
/// coarsely-split halves view) straight into the per-head ops must diverge.
void run_packed_gate_decode_tests(celeg::CudaStream& stream);
}
