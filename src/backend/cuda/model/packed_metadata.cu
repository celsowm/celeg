#include "celeg/backend/cuda/packed_metadata.hpp"

#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/detail/model/compiled_model.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace celeg {

void stage_packed_persistent_metadata(
    PackedWorkspace& workspace,
    const std::vector<PackedSessionContext>& models,
    const RuntimeTopology& shape) {
    const size_t rows = models.size();
    for (size_t row = 0; row < rows; ++row) {
        const PackedSessionContext& model = models[row];
        workspace.h_logits.data()[row] = model.logits().data();
        workspace.h_seen.data()[row] = model.seen_tokens().data();
        workspace.h_rng.data()[row] = model.rng_state().data();
        workspace.h_sampled_dest.data()[row] = model.sampled_device().data();
        workspace.h_position_dest.data()[row] = model.position_device().data();
        for (int layer_index = 0; layer_index < shape.num_hidden_layers; ++layer_index) {
            const size_t index = static_cast<size_t>(layer_index) *
                                 workspace.maximum_batch + row;
            Layer& layer = model.layers()[layer_index];
            if (auto* attention = as_attention(layer)) {
                workspace.h_key_bf16.data()[index] = attention->key_cache.data();
                workspace.h_value_bf16.data()[index] = attention->value_cache.data();
                workspace.h_key_int8.data()[index] = attention->key_cache_int8.data();
                workspace.h_value_int8.data()[index] = attention->value_cache_int8.data();
                workspace.h_key_scales.data()[index] = attention->key_cache_scales.data();
                workspace.h_value_scales.data()[index] = attention->value_cache_scales.data();
                workspace.h_conv_states.data()[index] = nullptr;
            } else {
                auto* convolution = as_convolution(layer);
                workspace.h_key_bf16.data()[index] = nullptr;
                workspace.h_value_bf16.data()[index] = nullptr;
                workspace.h_key_int8.data()[index] = nullptr;
                workspace.h_value_int8.data()[index] = nullptr;
                workspace.h_key_scales.data()[index] = nullptr;
                workspace.h_value_scales.data()[index] = nullptr;
                workspace.h_conv_states.data()[index] = convolution->conv_state.data();
            }
        }
    }

    const size_t pointer_count = rows * sizeof(void*);
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_logits.data(), workspace.h_logits.data(),
                               pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_seen.data(), workspace.h_seen.data(),
                               pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_rng.data(), workspace.h_rng.data(),
                               pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_sampled_dest.data(), workspace.h_sampled_dest.data(),
                               pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_position_dest.data(), workspace.h_position_dest.data(),
                               pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));

    const size_t layer_pointer_count =
        static_cast<size_t>(shape.num_hidden_layers) * workspace.maximum_batch * sizeof(void*);
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_key_bf16.data(), workspace.h_key_bf16.data(),
                               layer_pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_value_bf16.data(), workspace.h_value_bf16.data(),
                               layer_pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_key_int8.data(), workspace.h_key_int8.data(),
                               layer_pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_value_int8.data(), workspace.h_value_int8.data(),
                               layer_pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_key_scales.data(), workspace.h_key_scales.data(),
                               layer_pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_value_scales.data(), workspace.h_value_scales.data(),
                               layer_pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.d_conv_states.data(), workspace.h_conv_states.data(),
                               layer_pointer_count, cudaMemcpyHostToDevice,
                               workspace.stream.get()));
}

void stage_packed_step_metadata(
    PackedWorkspace& workspace,
    const std::vector<PackedSessionContext>& models,
    const std::vector<std::vector<uint32_t>>* page_tables) {
    const size_t rows = models.size();
    const int page_stride = workspace.paged_kv
        ? workspace.paged_kv->max_pages_per_request() : 0;
    if (workspace.paged_kv) {
        if (!page_tables || page_tables->size() != rows) {
            throw std::invalid_argument("paged packed decode requires one page table per row");
        }
        std::fill_n(workspace.h_page_tables.data(),
                    rows * static_cast<size_t>(page_stride),
                    std::numeric_limits<uint32_t>::max());
    }
    for (size_t row = 0; row < rows; ++row) {
        const PackedSessionContext& model = models[row];
        workspace.h_positions.data()[row] = model.position();
        workspace.h_temperatures.data()[row] = model.generation().temperature;
        workspace.h_repetition_penalties.data()[row] = model.generation().repetition_penalty;
        workspace.h_top_k.data()[row] = model.generation().top_k;
        workspace.h_top_p.data()[row] = model.generation().top_p;
        if (workspace.paged_kv) {
            const auto& pages = page_tables->at(row);
            const size_t needed = (static_cast<size_t>(model.position()) +
                static_cast<size_t>(workspace.paged_kv->page_tokens())) /
                static_cast<size_t>(workspace.paged_kv->page_tokens());
            if (pages.size() < needed || pages.size() > static_cast<size_t>(page_stride)) {
                throw std::invalid_argument("paged packed decode page table has invalid length");
            }
            std::copy(pages.begin(), pages.end(), workspace.h_page_tables.data() +
                      row * static_cast<size_t>(page_stride));
        }
    }
    const size_t scalar_i32 = rows * sizeof(int32_t);
    const size_t scalar_f32 = rows * sizeof(float);
    CELEG_CUDA(cudaMemcpyAsync(workspace.positions.data(), workspace.h_positions.data(),
                               scalar_i32, cudaMemcpyHostToDevice, workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.temperatures.data(), workspace.h_temperatures.data(),
                               scalar_f32, cudaMemcpyHostToDevice, workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.repetition_penalties.data(),
                               workspace.h_repetition_penalties.data(), scalar_f32,
                               cudaMemcpyHostToDevice, workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.top_k.data(), workspace.h_top_k.data(), scalar_i32,
                               cudaMemcpyHostToDevice, workspace.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace.top_p.data(), workspace.h_top_p.data(), scalar_f32,
                               cudaMemcpyHostToDevice, workspace.stream.get()));
    if (workspace.paged_kv) {
        const size_t page_bytes = rows * static_cast<size_t>(page_stride) * sizeof(uint32_t);
        CELEG_CUDA(cudaMemcpyAsync(workspace.d_page_tables.data(),
                                   workspace.h_page_tables.data(), page_bytes,
                                   cudaMemcpyHostToDevice, workspace.stream.get()));
    }
}

} // namespace celeg
