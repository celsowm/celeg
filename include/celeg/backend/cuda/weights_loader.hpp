#pragma once

#include "celeg/backend/cuda/utils.cuh"
#include "celeg/checkpoint/formats/safetensors.hpp"
#include "celeg/checkpoint/repositories/safetensors.hpp"
#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/detail/model/expert_weights.hpp"
#include "celeg/detail/model/layer_state.hpp"
#include "celeg/detail/model/shared_weights.hpp"
#include "celeg/model/weights/roles.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace celeg {

void dequantize_gguf_to_bf16(const HostTensorView& tensor,
                             std::vector<__nv_bfloat16>& out);

class WeightLoader {
public:
    static std::shared_ptr<SharedModelWeights> acquire(
        const std::string& model_path,
        WeightMode weight_mode,
        const std::string& residency_fingerprint);

    WeightLoader(std::shared_ptr<SharedModelWeights> weights,
                 WeightMode weight_mode);

    const __nv_bfloat16* load_weight(
        const IWeightRepository& repo,
        const std::string& name,
        std::vector<int64_t> expected = {});
    const __nv_bfloat16* load_rms_norm_weight(
        const IWeightRepository& repo,
        const std::string& name,
        std::vector<int64_t> expected,
        NormWeightKind weight_kind);

    const LinearWeight* load_linear_weight(
        const IWeightRepository& repo,
        const std::string& name,
        std::vector<int64_t> expected);

    const LinearWeight* load_concat_linear_weight(
        const IWeightRepository& repo,
        const std::string& synthetic_name,
        const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts);

    const ExpertLinearWeight* load_expert_linear_weight(
        const IWeightRepository& repo,
        const std::string& name,
        int experts, int rows_per_expert, int cols);

    /// Routed-expert loaders consume the resolved per-layer tensor names
    /// taken from the authoritative weight plan (see
    /// moe_expert_tensor_names()); they never construct checkpoint spellings
    /// or probe the repository for layout by name. Storage family (GGUF
    /// block-quantized, checkpoint-packed int4, BF16) is decided by tensor
    /// dtype/capability, not by spelling.
    const ExpertLinearWeight* load_moe_gate_up(
        const IWeightRepository& repo,
        const MoeExpertTensorNames& names,
        int num_experts, int moe_intermediate, int hidden);

    const ExpertLinearWeight* load_moe_down(
        const IWeightRepository& repo,
        const MoeExpertTensorNames& names,
        int num_experts, int moe_intermediate, int hidden);

    const float* load_f32_weight(
        const IWeightRepository& repo,
        const std::string& name,
        std::vector<int64_t> expected);

    const LinearWeight* load_router_weight_named(
        const IWeightRepository& repo, const std::string& name,
        int num_experts, int hidden);

    struct HostExpertLayer {
        std::vector<const __nv_bfloat16*> gate_up_host_dev;
        std::vector<const __nv_bfloat16*> down_host_dev;
        size_t gate_up_bytes = 0;
        size_t down_bytes = 0;
    };

    HostExpertLayer load_moe_experts_host(
        const IWeightRepository& repo,
        const MoeExpertTensorNames& names,
        int num_experts, int moe_intermediate, int hidden,
        class HostExpertStore& store, ExpertHostMode host_mode);

    std::vector<ExpertLocation> build_expert_catalog(
        const IWeightRepository& repo,
        const MoeExpertTensorNames& names,
        int num_experts, int moe_intermediate, int hidden);

    WeightMode weight_mode() const { return weight_mode_; }

private:
    std::shared_ptr<SharedModelWeights> weights_;
    WeightMode weight_mode_;
    std::unordered_map<std::string, ExpertLinearWeight> expert_cache_;
};

}
