#pragma once

#include "packed/metadata_cache.hpp"
#include "packed/workspace.hpp"

#include <cstdint>
#include <vector>

namespace celeg {

struct PackedAttentionBatchPlan {
    bool segmented = false;
    int chunks = 0;
};

class PackedAttentionBatchPlanner {
public:
    PackedAttentionBatchPlanner(PackedWorkspace& workspace,
                                PackedMetadataCache& metadata_cache)
        : workspace_(workspace), metadata_cache_(metadata_cache) {}

    PackedAttentionBatchPlan prepare_decode(
        const std::vector<PackedSessionContext>& models,
        const std::vector<std::vector<uint32_t>>* page_tables);
    PackedAttentionBatchPlan prepare_prefill(
        const std::vector<PackedSessionContext>& models,
        const std::vector<std::vector<uint32_t>>& page_tables,
        const std::vector<PackedPrefillRow>& descriptors);

private:
    PackedWorkspace& workspace_;
    PackedMetadataCache& metadata_cache_;
};

}
