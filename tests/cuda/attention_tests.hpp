#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
void run_attention_tests(celeg::CudaStream& stream);
/// Exercises attention data-movement helpers (packed gate extract, QKV split).
void run_attention_data_movement_tests(celeg::CudaStream& stream);
}
