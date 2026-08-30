#include "latent_path.hpp"

#include <stdexcept>

int main() {
    celeg::AttentionSpec factorized;
    celeg::LatentAttentionStateSpec factorized_state;
    factorized_state.projection = celeg::FactorizedLatentProjection{};
    factorized.state = factorized_state;
    if (celeg::prefill_detail::latent_prefill_path(factorized) !=
        celeg::prefill_detail::LatentPrefillPath::Factorized) {
        return 1;
    }

    celeg::AttentionSpec projected;
    celeg::LatentAttentionStateSpec projected_state;
    projected_state.projection = celeg::DirectLatentProjection{};
    projected.state = projected_state;
    if (celeg::prefill_detail::latent_prefill_path(projected) !=
        celeg::prefill_detail::LatentPrefillPath::Projected) {
        return 2;
    }

    celeg::AttentionSpec ordinary;
    try {
        (void)celeg::prefill_detail::latent_prefill_path(ordinary);
    } catch (const std::logic_error&) {
        return 0;
    }
    return 3;
}
