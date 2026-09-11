#include "model/runtime/quant_registry.hpp"
#include "support/assertions.hpp"

#include <string_view>

using celeg::MetalModelOptions;
using celeg::MetalNumericalPolicy;
using celeg::MetalPipelineCache;
using celeg::TensorRole;
using celeg::metal_model_detail::MetalLinearStorage;

int main() {
    MetalModelOptions options;
    options.numerical_policy = MetalNumericalPolicy::Fast;

    MetalPipelineCache cache;
    cache.tensor_matmul_q5k = true;
    cache.tensor_fast_q5k = true;

    const auto ordinary = celeg::quant_matvec_kernel(
        MetalLinearStorage::Q5K, 1024, 1024, options, nil);
    CELEG_TEST_CHECK(ordinary.name == std::string_view{"celeg_matvec_q5k"});

    const auto ffn_expansion = celeg::quant_matvec_kernel(
        MetalLinearStorage::Q5K, 8192, 1024, options, nil);
    CELEG_TEST_CHECK(ffn_expansion.name == std::string_view{"celeg_matvec_q5k"});
    CELEG_TEST_CHECK(ffn_expansion.rows_per_threadgroup == 16);
    CELEG_TEST_CHECK(ffn_expansion.threads == 128);

    CELEG_TEST_CHECK(celeg::quant_tensor_matmul_available(
        MetalLinearStorage::Q5K, cache));
    CELEG_TEST_CHECK(!celeg::quant_fast_tensor_matmul_available(
        MetalLinearStorage::Q5K, cache));

    const auto prefill = celeg::quant_select_tensor_kernel(
        MetalLinearStorage::Q5K,
        128,
        8192,
        1024,
        TensorRole::FfnUp,
        cache,
        options,
        nil);
    CELEG_TEST_CHECK(prefill.name == std::string_view{"celeg_matmul_tensor_q5k"});
    CELEG_TEST_CHECK(!prefill.custom);
    CELEG_TEST_CHECK(prefill.tile_tokens == 128);

    const auto short_prefill = celeg::quant_select_tensor_kernel(
        MetalLinearStorage::Q5K,
        16,
        8192,
        1024,
        TensorRole::FfnUp,
        cache,
        options,
        nil);
    CELEG_TEST_CHECK(short_prefill.name == std::string_view{"celeg_matmul_tensor_q5k"});
    CELEG_TEST_CHECK(!short_prefill.custom);
    CELEG_TEST_CHECK(short_prefill.tile_tokens == 128);

    return 0;
}
