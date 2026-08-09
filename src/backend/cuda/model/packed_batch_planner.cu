#include "celeg/backend/cuda/packed_batch_planner.hpp"

#include "celeg/backend/cuda/packed_metadata.hpp"
#include "celeg/backend/cuda/paged_kv.hpp"

#include <algorithm>
#include <limits>

namespace celeg {

PackedAttentionBatchPlan PackedAttentionBatchPlanner::prepare_decode(
    const std::vector<PackedSessionContext>& models,
    const std::vector<std::vector<uint32_t>>* page_tables) {
    if (metadata_cache_.changed(models)) {
        stage_packed_persistent_metadata(workspace_, models, workspace_.shape_);
        metadata_cache_.update(models);
    }
    stage_packed_step_metadata(workspace_, models, page_tables);

    PackedAttentionBatchPlan plan;
    int maximum_position = 0;
    for (size_t row = 0; row < models.size(); ++row) {
        maximum_position = std::max(maximum_position,
                                    workspace_.h_positions.data()[row]);
        plan.segmented = plan.segmented ||
            models[row].use_segmented_attention(
                workspace_.h_positions.data()[row]);
    }
    if (plan.segmented) {
        const int chunk_tokens = models.front().options().attention_chunk_tokens;
        plan.chunks = (maximum_position + 1 + chunk_tokens - 1) / chunk_tokens;
        workspace_.ensure_segmented_workspace(static_cast<int>(models.size()),
                                               plan.chunks);
    }
    return plan;
}

PackedAttentionBatchPlan PackedAttentionBatchPlanner::prepare_prefill(
    const std::vector<PackedSessionContext>& models,
    const std::vector<std::vector<uint32_t>>& page_tables,
    const std::vector<PackedPrefillRow>& descriptors) {
    const size_t tokens = [&] {
        size_t value = 0;
        for (const PackedPrefillRow& row : descriptors) value += row.token_count;
        return value;
    }();
    const int page_stride = workspace_.paged_kv->max_pages_per_request();
    std::fill_n(workspace_.h_page_tables.data(),
                tokens * static_cast<size_t>(page_stride),
                std::numeric_limits<uint32_t>::max());
    size_t flat = 0;
    int maximum_position = 0;
    for (size_t request = 0; request < models.size(); ++request) {
        const int start = models[request].position();
        const auto& pages = page_tables[request];
        for (size_t token = 0; token < descriptors[request].token_count;
             ++token, ++flat) {
            workspace_.h_positions.data()[flat] = start + static_cast<int>(token);
            maximum_position = std::max(maximum_position,
                                        workspace_.h_positions.data()[flat]);
            std::copy(pages.begin(), pages.end(), workspace_.h_page_tables.data() +
                flat * static_cast<size_t>(page_stride));
        }
    }
    CELEG_CUDA(cudaMemcpyAsync(workspace_.positions.data(),
                               workspace_.h_positions.data(),
                               tokens * sizeof(int32_t), cudaMemcpyHostToDevice,
                               workspace_.stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace_.d_page_tables.data(),
                               workspace_.h_page_tables.data(),
                               tokens * static_cast<size_t>(page_stride) *
                                   sizeof(uint32_t),
                               cudaMemcpyHostToDevice, workspace_.stream.get()));

    PackedAttentionBatchPlan plan;
    for (size_t request = 0; request < models.size(); ++request) {
        plan.segmented = plan.segmented || models[request].use_segmented_attention(
            models[request].position() +
            static_cast<int>(descriptors[request].token_count) - 1);
    }
    if (plan.segmented) {
        const int chunk_tokens = models.front().options().attention_chunk_tokens;
        plan.chunks = (maximum_position + 1 + chunk_tokens - 1) / chunk_tokens;
        workspace_.ensure_segmented_workspace(static_cast<int>(tokens), plan.chunks);
    }
    return plan;
}

} // namespace celeg
