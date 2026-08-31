#pragma once

#include "detail/compiled_model.hpp"

namespace celeg {

void require_cuda_token_latent_attention_paged(
    const CudaCompiledModel& model, const AttentionLayer& attention);

void execute_cuda_token_latent_attention_paged(
    CudaCompiledModel& model, AttentionLayer& attention, int layer_index,
    const CudaCompiledModel::TokenKvPolicy& kv);

}
