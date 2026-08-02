#include "celeg/detail/model/compiled_model.hpp"

#include <cmath>
#include <vector>
namespace celeg {

void CudaCompiledModel::initialize_rope_tables() {
    const size_t table_elements =
        static_cast<size_t>(max_context_) * resources_.shape_.rope_pairs;
    std::vector<__nv_bfloat16> cos_table(table_elements);
    std::vector<__nv_bfloat16> sin_table(table_elements);

    for (int position = 0; position < max_context_; ++position) {
        for (int pair = 0; pair < resources_.shape_.rope_pairs; ++pair) {
            const float exponent =
                -2.0f * static_cast<float>(pair) /
                static_cast<float>(resources_.shape_.head_dim);
            const float frequency = std::pow(resources_.shape_.rope_theta, exponent);
            const float angle = static_cast<float>(position) * frequency;
            const size_t index =
                static_cast<size_t>(position) * resources_.shape_.rope_pairs + pair;
            cos_table[index] = __float2bfloat16(std::cos(angle));
            sin_table[index] = __float2bfloat16(std::sin(angle));
        }
    }

    workspace_.rope_cos_.reset(table_elements);
    workspace_.rope_sin_.reset(table_elements);
    CELEG_CUDA(cudaMemcpy(workspace_.rope_cos_.data(), cos_table.data(),
                        cos_table.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(workspace_.rope_sin_.data(), sin_table.data(),
                        sin_table.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
}

} // namespace celeg

