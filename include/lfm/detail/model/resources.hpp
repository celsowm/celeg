#pragma once

#include "lfm/model/config/shape.hpp"
#include "lfm/model/config/variant.hpp"
#include "lfm/model/execution/plan.hpp"
#include "lfm/model/weights/layout.hpp"
#include "lfm/model/weights/loader.hpp"
#include "lfm/model/weights/roles.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace lfm {

// CUDA model resource owner for immutable/shared topology decisions. Device
// weights and backend caches are attached by setup, while request-local
// position, phase, and metrics live in SessionState.
struct ModelResources {
    explicit ModelResources(ExecutionPlan execution_plan)
        : plan_(std::move(execution_plan)), options_(plan_.options()) {}

    ExecutionPlan plan_;
    ModelOptions options_;
    ModelShape shape_;
    const IModelVariant* variant_ = nullptr;
    const ITensorNamingPolicy* tensor_naming_ = nullptr;
    std::shared_ptr<SharedModelWeights> weights_;
    std::unique_ptr<WeightLoader> weight_loader_;
    std::vector<Layer> layers_;
    const LinearWeight* embedding_ = nullptr;
    const LinearWeight* lm_head_ = nullptr;
    const __nv_bfloat16* final_norm_ = nullptr;
    std::unique_ptr<IWeightLayout> weight_layout_;
};

} // namespace lfm
