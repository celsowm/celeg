#include "lfm/expert_offload.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace lfm {

namespace {

// BF16 element size. Kept as a plain constant so this host translation unit
// does not depend on the CUDA toolkit headers (__nv_bfloat16).
constexpr std::size_t kBf16Bytes = 2;

std::string format_bytes(std::size_t bytes) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 3) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2)
        << value << ' ' << units[unit];
    return out.str();
}

const char* host_mode_name(ExpertHostMode mode) {
    switch (mode) {
        case ExpertHostMode::Mapped: return "mapped";
        case ExpertHostMode::PinnedCopy: return "pinned-copy";
        case ExpertHostMode::Staged: return "staged";
    }
    return "unknown";
}

const char* policy_name(ExpertCachePolicy policy) {
    switch (policy) {
        case ExpertCachePolicy::Static: return "static";
        case ExpertCachePolicy::Lru: return "lru";
        case ExpertCachePolicy::LayerLocalLfuLru: return "lfu-lru";
    }
    return "unknown";
}

} // namespace

std::size_t bytes_per_expert_bf16(const ModelShape& shape) {
    const std::size_t gate_up = static_cast<std::size_t>(2) *
        static_cast<std::size_t>(shape.moe_intermediate) *
        static_cast<std::size_t>(shape.hidden);
    const std::size_t down = static_cast<std::size_t>(shape.hidden) *
        static_cast<std::size_t>(shape.moe_intermediate);
    return (gate_up + down) * kBf16Bytes;
}

int moe_layer_count(const ModelShape& shape) {
    if (shape.architecture != ArchitectureKind::MoeLfm2) return 0;
    return std::max(0, shape.num_hidden_layers - shape.num_dense_layers);
}

std::size_t kv_cache_bytes(const ModelShape& shape, int context_tokens) {
    if (context_tokens <= 0) return 0;
    const std::size_t per_token =
        static_cast<std::size_t>(2) *
        static_cast<std::size_t>(shape.num_key_value_heads) *
        static_cast<std::size_t>(shape.head_dim) *
        kBf16Bytes;
    return per_token *
        static_cast<std::size_t>(shape.attention_layer_count) *
        static_cast<std::size_t>(context_tokens);
}

ExpertOffloadPlan plan_expert_offload(const ExpertOffloadPlanInputs& inputs) {
    const ModelShape& shape = inputs.shape;
    const ExpertOffloadOptions& options = inputs.options;

    ExpertOffloadPlan plan;
    plan.moe_layers = moe_layer_count(shape);
    plan.bytes_per_expert = bytes_per_expert_bf16(shape);
    plan.gpu_free_bytes = inputs.gpu_free_bytes;
    plan.non_expert_weight_bytes = inputs.non_expert_weight_bytes;
    plan.kv_reservation_bytes = kv_cache_bytes(shape, inputs.context_tokens);
    plan.workspace_bytes = inputs.workspace_bytes;
    plan.reserve_bytes = options.gpu_memory_reserve_bytes;
    plan.host_mode = options.host_mode;
    plan.policy = options.policy;

    if (!options.enabled() || plan.moe_layers == 0 || shape.num_experts <= 0) {
        plan.enabled = false;
        plan.experts_per_layer = shape.num_experts;
        plan.host_experts_per_layer = 0;
        plan.gpu_expert_cache_bytes =
            static_cast<std::size_t>(shape.num_experts) *
            static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert;
        return plan;
    }

    plan.enabled = true;

    int slots_per_layer = 0;
    if (options.experts_per_layer > 0) {
        slots_per_layer = options.experts_per_layer;
    } else if (options.gpu_expert_cache_bytes > 0) {
        slots_per_layer = static_cast<int>(
            options.gpu_expert_cache_bytes /
            (static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert));
    } else {
        // Auto: derive the cache budget from free VRAM.
        const std::size_t fixed = plan.non_expert_weight_bytes +
            plan.kv_reservation_bytes + plan.workspace_bytes + plan.reserve_bytes;
        std::size_t available_for_experts = 0;
        if (plan.gpu_free_bytes > fixed) {
            available_for_experts = plan.gpu_free_bytes - fixed;
        }
        slots_per_layer = static_cast<int>(
            available_for_experts /
            (static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert));
    }

    const int min_experts =
        std::max(0, std::min(options.minimum_experts_per_layer, shape.num_experts));
    if (slots_per_layer < min_experts) {
        std::ostringstream msg;
        msg << "expert offload cannot fit the minimum "
            << min_experts << " experts/layer on the GPU (room for "
            << slots_per_layer << ")";
        throw std::runtime_error(msg.str());
    }
    slots_per_layer = std::min(slots_per_layer, shape.num_experts);

    // Respect the host pinned budget: every non-resident expert may need host
    // storage. If it would overflow, force more experts onto the GPU.
    if (options.host_mode == ExpertHostMode::PinnedCopy &&
        options.maximum_pinned_host_bytes > 0) {
        const std::size_t per_layer_host_bytes =
            static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert;
        const int max_host_per_layer = static_cast<int>(
            options.maximum_pinned_host_bytes / per_layer_host_bytes);
        const int min_gpu_for_host = shape.num_experts - max_host_per_layer;
        slots_per_layer = std::max(slots_per_layer, min_gpu_for_host);
        slots_per_layer = std::min(slots_per_layer, shape.num_experts);
    }

    plan.experts_per_layer = slots_per_layer;
    plan.host_experts_per_layer = shape.num_experts - slots_per_layer;
    // Prefetch depth cannot exceed the experts that are actually host-resident
    // (no point prefetching what is already cached).
    plan.prefetch_experts = std::max(
        0, std::min(options.prefetch_experts, plan.host_experts_per_layer));
    plan.gpu_expert_cache_bytes =
        static_cast<std::size_t>(slots_per_layer) *
        static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert;
    plan.host_expert_bytes =
        static_cast<std::size_t>(plan.host_experts_per_layer) *
        static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert;
    return plan;
}

std::string ExpertOffloadPlan::report() const {
    std::ostringstream out;
    out << "MoE offload plan:\n";
    if (!enabled) {
        out << "  enabled:                  no (all experts GPU-resident)\n"
            << "  experts per layer:        " << experts_per_layer << " / "
            << experts_per_layer << '\n';
        return out.str();
    }
    out << "  GPU memory free:          " << format_bytes(gpu_free_bytes) << '\n'
        << "  non-expert weights:       " << format_bytes(non_expert_weight_bytes) << '\n'
        << "  KV reservation:           " << format_bytes(kv_reservation_bytes) << '\n'
        << "  workspace/headroom:       "
        << format_bytes(workspace_bytes + reserve_bytes) << '\n'
        << "  GPU expert cache:         " << format_bytes(gpu_expert_cache_bytes) << '\n'
        << "  experts per layer:        " << experts_per_layer << " / "
        << (experts_per_layer + host_experts_per_layer) << '\n'
        << "  host expert storage:      " << format_bytes(host_expert_bytes) << '\n'
        << "  bytes per expert:         " << format_bytes(bytes_per_expert) << '\n'
        << "  host mode:                " << host_mode_name(host_mode) << '\n'
        << "  cache policy:             " << policy_name(policy) << '\n'
        << "  prefetch depth:           " << prefetch_experts << '\n';
    return out.str();
}

} // namespace lfm
