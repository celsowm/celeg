#include "cuda/attention_tests.hpp"
#include "cuda/sampling_tests.hpp"
#include "embedding_tests.hpp"
#include "gated_delta_tests.hpp"
#include "gemm_dispatcher_tests.hpp"
#include "mamba_tests.hpp"
#include "misc_tests.hpp"
#include "norm_activation_tests.hpp"
#include "paged_cache_tests.hpp"
#include "paged_decode_tests.hpp"
#include "quantized_linear_tests.hpp"
#include "utils.cuh"

#include <iostream>

int main() {
    celeg::CudaStream stream;

    celeg::cuda_test::run_mamba_tests(stream);
    celeg::cuda_test::run_gated_delta_tests(stream);
    celeg::cuda_test::run_attention_data_movement_tests(stream);
    celeg::cuda_test::run_embedding_tests(stream);
    celeg::cuda_test::run_quantized_linear_tests(stream);
    celeg::cuda_test::run_norm_activation_tests(stream);
    celeg::cuda_test::run_attention_tests(stream);
    celeg::cuda_test::run_sampling_tests(stream);
    celeg::cuda_test::run_gemm_dispatcher_tests(stream);
    celeg::cuda_test::run_paged_decode_tests(stream);
    celeg::cuda_test::run_packed_sampling_tests(stream);
    celeg::cuda_test::run_paged_cache_tests(stream);
    celeg::cuda_test::run_misc_tests(stream);

    std::cout << "cuda_kernels_test: ok\n";
    return 0;
}
