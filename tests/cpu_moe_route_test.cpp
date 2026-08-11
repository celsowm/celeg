#include "operators/moe.hpp"

#include "support/assertions.hpp"

#include <vector>

int main() {
    celeg::RouterProgram program;
    program.score = celeg::MoeRouterScoreKind::SoftmaxLogits;
    program.normalization = celeg::MoeNormalizationKind::SumSelected;
    program.expert_count = 3;
    program.experts_per_token = 2;
    program.routed_scaling = 0.5f;
    program.validate();

    const std::vector<float> logits{1.0f, 0.0f, 2.0f};
    const std::vector<float> bias{0.0f, -10.0f, 0.0f};
    const celeg::CpuMoeRoute route = celeg::route_cpu_moe(program, logits, bias);
    CELEG_TEST_CHECK(route.experts.size() == 2);
    CELEG_TEST_CHECK(route.experts[0] == 2 && route.experts[1] == 0);
    CELEG_TEST_CHECK(route.weights.size() == 2);
    CELEG_TEST_CHECK(route.weights[0] > route.weights[1]);
    CELEG_TEST_CHECK(route.weights[0] + route.weights[1] > 0.49f &&
                     route.weights[0] + route.weights[1] < 0.51f);

    celeg::RouterProgram grouped;
    grouped.score = celeg::MoeRouterScoreKind::SigmoidProbabilities;
    grouped.selection = celeg::MoeSelectionKind::GroupedTopK;
    grouped.normalization = celeg::MoeNormalizationKind::SumSelected;
    grouped.expert_count = 6;
    grouped.experts_per_token = 2;
    grouped.group_count = 2;
    grouped.experts_per_group = 3;
    grouped.groups_per_token = 1;
    grouped.group_score_top_k = 2;
    grouped.validate();
    const std::vector<float> grouped_logits{5.0f, 4.0f, -5.0f, 3.0f, 3.0f, 3.0f};
    const std::vector<float> no_group_bias;
    const celeg::CpuMoeRoute grouped_route = celeg::route_cpu_moe(
        grouped, grouped_logits, no_group_bias);
    CELEG_TEST_CHECK(grouped_route.experts.size() == 2);
    CELEG_TEST_CHECK(grouped_route.experts[0] == 0 && grouped_route.experts[1] == 1);
}
