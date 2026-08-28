#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/attention_norm.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

void CudaCompiledModel::run_token_mlp_only(MlpOnlyLayer& mlp) {
    linear(workspace_.normed_.data(), *mlp.up, workspace_.gate_up_.data(),
           1, mlp.spec.intermediate_size, resources_.program_.hidden);
    launch_relu2(workspace_.gate_up_.data(), workspace_.activated_.data(),
                 mlp.spec.intermediate_size, stream_.get());
    linear(workspace_.activated_.data(), *mlp.down, workspace_.hidden_.data(),
           1, resources_.program_.hidden, mlp.spec.intermediate_size);
}

void CudaCompiledModel::run_token_convolution(ConvolutionLayer& convolution) {
    linear(workspace_.normed_.data(), *convolution.conv_in,
           workspace_.conv_projected_.data(),
           1, 3 * resources_.program_.hidden, resources_.program_.hidden);
    launch_conv_decode(
        workspace_.conv_projected_.data(), convolution.conv_weight,
        convolution.conv_state.data(), workspace_.op_output_.data(),
        resources_.program_.hidden, convolution.spec.cache_length, session_.position_,
        stream_.get());
    linear(workspace_.op_output_.data(), *convolution.conv_out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, resources_.program_.hidden);
}

}
