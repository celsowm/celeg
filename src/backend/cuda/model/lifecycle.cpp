#include "detail/compiled_model.hpp"

#include <cuda_runtime.h>
namespace celeg {

CudaCompiledModel::~CudaCompiledModel() {
    if (resources_.weights_ && !resources_.weights_->usage_profile_path.empty()) {
        resources_.weights_->usage_stats.save(resources_.weights_->usage_profile_path);
    }
    if (workspace_.moe_sel_host_ != nullptr) {
        cudaFreeHost(workspace_.moe_sel_host_);
        workspace_.moe_sel_host_ = nullptr;
    }
    if (workspace_.moe_route_scores_host_ != nullptr) {
        cudaFreeHost(workspace_.moe_route_scores_host_);
        workspace_.moe_route_scores_host_ = nullptr;
    }
}

}
