#include "celeg/backend/cpu/kv_topology.hpp"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace celeg {
namespace {

const CompiledAttentionProgram* attention_program(
    const CompiledLayerProgram& layer) {
    return std::get_if<CompiledAttentionProgram>(&layer.mixer);
}

} // namespace

CpuStatePageLayout lower_cpu_state_page_layout(
    const CompiledAttentionStateLayout& state) {
    return std::visit([](const auto& layout) -> CpuStatePageLayout {
        using Layout = std::decay_t<decltype(layout)>;
        if constexpr (std::is_same_v<Layout, CompiledOrdinaryKvStateLayout>) {
            return {
                static_cast<std::size_t>(layout.key_width),
                static_cast<std::size_t>(layout.value_width), 0, 0};
        } else if constexpr (std::is_same_v<Layout, CompiledLatentStateLayout>) {
            return {
                0, 0,
                static_cast<std::size_t>(layout.latent_width),
                static_cast<std::size_t>(layout.rotary_width)};
        } else {
            static_assert(always_false_v<Layout>,
                          "unhandled CPU state-page layout");
        }
    }, state);
}

CpuKvTopology build_cpu_kv_topology(const ExecutionTopology& shape,
                                    const CompiledModelProgram& program,
                                    const CpuModelOptions& options) {
    CpuKvTopology result;
    result.layer_to_pool.assign(static_cast<size_t>(shape.num_hidden_layers), -1);
    result.layer_to_owner.assign(static_cast<size_t>(shape.num_hidden_layers), -1);

    int shared_group_count = 0;
    for (const CompiledLayerProgram& layer : program.layers) {
        if (const auto* compiled = attention_program(layer);
            compiled && compiled->semantics.kv_sharing.shared()) {
            shared_group_count = std::max(
                shared_group_count, compiled->semantics.kv_sharing.group + 1);
        }
    }
    std::vector<int> shared_owner(static_cast<size_t>(shared_group_count), -1);
    std::vector<int> shared_pool(static_cast<size_t>(shared_group_count), -1);

    for (int layer = 0; layer < shape.num_hidden_layers; ++layer) {
        if (layer < 0 || layer >= static_cast<int>(program.layers.size())) continue;
        const CompiledLayerProgram& compiled =
            program.layers[static_cast<size_t>(layer)];
        const auto* attention = attention_program(compiled);
        if (!attention) continue;
        if (attention->semantics.kv_sharing.publishes) {
            shared_owner[static_cast<size_t>(attention->semantics.kv_sharing.group)] = layer;
        }
    }

    for (int layer = 0; layer < shape.num_hidden_layers; ++layer) {
        if (layer < 0 || layer >= static_cast<int>(program.layers.size())) continue;
        const size_t index = static_cast<size_t>(layer);
        const CompiledLayerProgram& compiled = program.layers[index];
        const auto* attention = attention_program(compiled);
        if (!attention) continue;
        const CpuStatePageLayout state_layout =
            lower_cpu_state_page_layout(attention->state_layout);
        if (attention->semantics.kv_sharing.shared() &&
            !attention->semantics.kv_sharing.publishes) {
            const int group = attention->semantics.kv_sharing.group;
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
            options.kv_cache_mode, options.kv_page_tokens, state_layout));
        if (attention->semantics.kv_sharing.shared()) {
            shared_pool[static_cast<size_t>(attention->semantics.kv_sharing.group)] =
                result.layer_to_pool[index];
            result.layer_to_owner[index] = layer;
        } else {
            result.layer_to_owner[index] = layer;
        }
    }
    return result;
}

} // namespace celeg
