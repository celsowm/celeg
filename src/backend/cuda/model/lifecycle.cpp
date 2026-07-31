#include "celeg/detail/model/impl.hpp"

#include <cuda_runtime.h>
namespace celeg {

Model::Impl::~Impl() {
    if (weights_ && !weights_->usage_profile_path.empty()) {
        weights_->usage_stats.save(weights_->usage_profile_path);
    }
    if (moe_sel_host_ != nullptr) {
        cudaFreeHost(moe_sel_host_);
        moe_sel_host_ = nullptr;
    }
    if (moe_route_scores_host_ != nullptr) {
        cudaFreeHost(moe_route_scores_host_);
        moe_route_scores_host_ = nullptr;
    }
}

} // namespace celeg

