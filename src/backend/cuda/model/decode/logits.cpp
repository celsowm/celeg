#include "detail/compiled_model.hpp"
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

void CudaCompiledModel::run_token_logits() {
    launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_, workspace_.normed_.data(),
                   1, resources_.program_.hidden, resources_.program_.final_norm.epsilon,
                   stream_.get());
    linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
           1, resources_.dims_.vocab_size, resources_.program_.hidden);
    launch_scale(workspace_.logits_.data(), resources_.dims_.vocab_size,
                 resources_.program_.logits_multiplier /
                     resources_.program_.logits_divisor, stream_.get());
    if (resources_.program_.final_logit_softcap > 0.0f) {
        launch_tanh_softcap(workspace_.logits_.data(), resources_.dims_.vocab_size,
                            resources_.program_.final_logit_softcap, stream_.get());
    }
    launch_mask_logits(workspace_.logits_.data(), resources_.dims_.vocab_size,
                       tokenizer_vocab_size_, stream_.get());
}

}
