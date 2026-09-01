#include "celeg/model/graph.hpp"
#include "support/assertions.hpp"

#include <stdexcept>
#include <utility>

namespace {

celeg::ModelGraph graph_with_bias(celeg::AttentionBiasSpec bias) {
    celeg::ModelGraph graph;
    graph.hidden = 8;

    celeg::AttentionSpec attention;
    attention.query_heads = 2;
    attention.key_value_heads = 2;
    attention.head_dim = 4;
    attention.position = celeg::NoPositionEncodingSpec{};
    attention.bias = std::move(bias);

    celeg::LayerSpec layer;
    layer.mixer = std::move(attention);
    layer.feed_forward = std::monostate{};
    graph.layers.push_back(std::move(layer));
    return graph;
}

bool graph_rejects(celeg::AttentionBiasSpec bias) {
    celeg::ModelGraph graph = graph_with_bias(std::move(bias));
    try {
        graph.validate();
    } catch (const std::runtime_error&) {
        return true;
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    CELEG_TEST_CHECK(!graph_rejects(celeg::NoAttentionBiasSpec{}));
    CELEG_TEST_CHECK(!graph_rejects(celeg::AlibiBiasSpec{{0.5f, 1.0f}}));
    CELEG_TEST_CHECK(graph_rejects(celeg::AlibiBiasSpec{{0.5f}}));
    CELEG_TEST_CHECK(graph_rejects(celeg::AlibiBiasSpec{{0.5f, -1.0f}}));

    CELEG_TEST_CHECK(!graph_rejects(
        celeg::RelativePositionBiasSpec{2, 16, false}));
    CELEG_TEST_CHECK(!graph_rejects(
        celeg::RelativePositionBiasSpec{4, 16, true}));

    CELEG_TEST_CHECK(graph_rejects(
        celeg::RelativePositionBiasSpec{1, 16, false}));
    CELEG_TEST_CHECK(graph_rejects(
        celeg::RelativePositionBiasSpec{2, 16, true}));
    CELEG_TEST_CHECK(graph_rejects(
        celeg::RelativePositionBiasSpec{3, 16, true}));
    CELEG_TEST_CHECK(graph_rejects(
        celeg::RelativePositionBiasSpec{4, 0, true}));

    return 0;
}
