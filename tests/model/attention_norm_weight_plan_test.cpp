#include "celeg/model/weight_plan.hpp"
#include "support/assertions.hpp"

#include <vector>

namespace {

class Naming final : public celeg::ITensorNamingPolicy {
public:
    std::vector<std::string> candidates(
        const celeg::TensorRequest&) const override {
        return {};
    }
};

celeg::AttentionSpec external_attention(celeg::NormSpec query_norm) {
    celeg::AttentionSpec attention;
    attention.query_heads = 2;
    attention.key_value_heads = 2;
    attention.head_dim = 4;
    attention.key_value_source = celeg::ExternalMemorySource{1};
    attention.pattern = celeg::BidirectionalPattern{};
    attention.position = celeg::NoPositionEncodingSpec{};
    attention.query_norm = query_norm;
    return attention;
}

bool has_request(const celeg::ResolvedModel& model, celeg::TensorRole role,
                 int layer) {
    for (const auto& request : model.weight_plan.requests) {
        if (request.role == role && request.layer == layer) return true;
    }
    return false;
}

const celeg::TensorRequest& request_for(const celeg::ResolvedModel& model,
                                       celeg::TensorRole role, int layer) {
    for (const auto& request : model.weight_plan.requests) {
        if (request.role == role && request.layer == layer) return request;
    }
    throw std::runtime_error("attention norm request missing");
}

}

int main() {
    celeg::ResolvedModel model;
    model.graph.hidden = 8;
    model.topology.dims.vocab_size = 32;

    celeg::LayerSpec weighted_external;
    weighted_external.mixer = external_attention(celeg::NormSpec{
        1.0e-5f, celeg::NormWeightKind::Scale,
        celeg::NormGranularity::WholeVector});
    weighted_external.feed_forward = std::monostate{};
    model.graph.layers.push_back(std::move(weighted_external));

    celeg::LayerSpec weightless_external;
    weightless_external.mixer = external_attention(celeg::NormSpec{
        1.0e-5f, celeg::NormWeightKind::None,
        celeg::NormGranularity::PerHead});
    weightless_external.feed_forward = std::monostate{};
    model.graph.layers.push_back(std::move(weightless_external));

    celeg::LayerSpec weightless_ordinary;
    celeg::AttentionSpec ordinary;
    ordinary.query_heads = 2;
    ordinary.key_value_heads = 2;
    ordinary.head_dim = 4;
    ordinary.query_norm = celeg::NormSpec{
        1.0e-5f, celeg::NormWeightKind::None,
        celeg::NormGranularity::PerHead};
    ordinary.key_norm = celeg::NormSpec{
        1.0e-5f, celeg::NormWeightKind::None,
        celeg::NormGranularity::WholeVector};
    weightless_ordinary.mixer = ordinary;
    weightless_ordinary.feed_forward = std::monostate{};
    model.graph.layers.push_back(std::move(weightless_ordinary));

    model.topology = celeg::compose_runtime_topology(
        celeg::CheckpointDimensions{32, 0, {}, {}, 0}, model.graph);

    Naming naming;
    celeg::build_weight_plan_from_graph(model, naming, nullptr);

    CELEG_TEST_CHECK(has_request(model, celeg::TensorRole::AttentionQuery, 0));
    CELEG_TEST_CHECK(has_request(model, celeg::TensorRole::AttentionOutput, 0));
    CELEG_TEST_CHECK(!has_request(model, celeg::TensorRole::AttentionKey, 0));
    CELEG_TEST_CHECK(!has_request(model, celeg::TensorRole::AttentionValue, 0));
    CELEG_TEST_CHECK(!has_request(model, celeg::TensorRole::AttentionKeyNorm, 0));
    const auto& weighted = request_for(
        model, celeg::TensorRole::AttentionQueryNorm, 0);
    CELEG_TEST_CHECK(weighted.expected_shape == std::vector<int64_t>{8});
    CELEG_TEST_CHECK(weighted.norm_weight_kind == celeg::NormWeightKind::Scale);

    CELEG_TEST_CHECK(!has_request(
        model, celeg::TensorRole::AttentionQueryNorm, 1));
    CELEG_TEST_CHECK(!has_request(
        model, celeg::TensorRole::AttentionQueryNorm, 2));
    CELEG_TEST_CHECK(!has_request(
        model, celeg::TensorRole::AttentionKeyNorm, 2));

    return 0;
}
