#include "celeg/backend/cpu/kv_topology.hpp"

#include <algorithm>
#include <stdexcept>

namespace celeg {

CpuKvTopology build_cpu_kv_topology(const RuntimeTopology& shape,
                                    const CpuModelOptions& options) {
    CpuKvTopology result;
    result.layer_to_pool.assign(static_cast<size_t>(shape.num_hidden_layers), -1);
    result.layer_to_owner.assign(static_cast<size_t>(shape.num_hidden_layers), -1);

    int shared_group_count = 0;
    for (const AttentionSpec& attention : shape.attention_layouts) {
        if (attention.kv_sharing.shared()) {
            shared_group_count = std::max(shared_group_count,
                                          attention.kv_sharing.group + 1);
        }
    }
    std::vector<int> shared_owner(static_cast<size_t>(shared_group_count), -1);
    std::vector<int> shared_pool(static_cast<size_t>(shared_group_count), -1);

    for (int layer = 0; layer < shape.num_hidden_layers; ++layer) {
        if (shape.mixer_kinds[static_cast<size_t>(layer)] != MixerKind::Attention) {
            continue;
        }
        const AttentionSpec& attention = shape.attention_layout(layer);
        if (attention.kv_sharing.publishes) {
            shared_owner[static_cast<size_t>(attention.kv_sharing.group)] = layer;
        }
    }

    for (int layer = 0; layer < shape.num_hidden_layers; ++layer) {
        if (shape.mixer_kinds[static_cast<size_t>(layer)] != MixerKind::Attention) {
            continue;
        }
        const size_t index = static_cast<size_t>(layer);
        const AttentionSpec& attention = shape.attention_layout(layer);
        if (attention.kv_sharing.shared() && !attention.kv_sharing.publishes) {
            const int group = attention.kv_sharing.group;
            if (group < 0 || group >= static_cast<int>(shared_pool.size()) ||
                shared_pool[static_cast<size_t>(group)] < 0) {
                throw std::runtime_error("shared KV consumer has no owner pool");
            }
            result.layer_to_pool[index] = shared_pool[static_cast<size_t>(group)];
            result.layer_to_owner[index] = shared_owner[static_cast<size_t>(group)];
            continue;
        }

        result.layer_to_pool[index] = static_cast<int>(result.pools.size());
        result.pools.push_back(std::make_shared<CpuKvPagePool>(
            options.kv_cache_mode, options.kv_page_tokens,
            static_cast<size_t>(attention.key_value_width())));
        if (attention.kv_sharing.shared()) {
            shared_pool[static_cast<size_t>(attention.kv_sharing.group)] =
                result.layer_to_pool[index];
            result.layer_to_owner[index] = layer;
        } else {
            result.layer_to_owner[index] = layer;
        }
    }
    return result;
}

} // namespace celeg
