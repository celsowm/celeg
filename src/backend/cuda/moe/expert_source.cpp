#include "celeg/backend/cuda/moe/expert_source.hpp"

#include "celeg/detail/model/shared_weights.hpp"
#include "celeg/backend/cuda/weights_loader.hpp"

#include <stdexcept>

namespace celeg {

void CudaExpertSource::read(const ExpertPayloadRequest& request,
                            std::span<std::byte> destination) const {
    request.validate();
    read(request.key.layer, request.key.expert, destination);
}

void CudaExpertSource::read(int layer, int expert,
                            std::span<std::byte> destination) const {
    if (!weights_ || layer < 0 || expert < 0 ||
        static_cast<size_t>(layer) >= weights_->expert_catalog.size() ||
        static_cast<size_t>(expert) >= weights_->expert_catalog[static_cast<size_t>(layer)].size()) {
        throw std::invalid_argument("expert source key is outside the catalog");
    }
    const ExpertLocation& location =
        weights_->expert_catalog[static_cast<size_t>(layer)][static_cast<size_t>(expert)];
    const size_t gate_up_bytes = location.w1.bytes + location.w3.bytes;
    const size_t required_bytes = gate_up_bytes + location.w2.bytes;
    if (destination.size() < required_bytes) {
        throw std::invalid_argument("expert source destination does not fit catalog entry");
    }

    std::span<std::byte> gate_up(destination.data(), gate_up_bytes);
    std::span<std::byte> down(destination.data() + gate_up_bytes, location.w2.bytes);
    if (weights_->expert_sidecar) {
        weights_->expert_sidecar->read_expert(layer, expert, gate_up, down);
        return;
    }
    if (!weights_->repo) {
        throw std::runtime_error("expert source has no checkpoint repository");
    }
    const auto& reader = require_random_access_tensor_reader(*weights_->repo);
    reader.read(location.w1, gate_up.subspan(0, location.w1.bytes));
    reader.read(location.w3, gate_up.subspan(location.w1.bytes, location.w3.bytes));
    reader.read(location.w2, down);
}

} // namespace celeg
