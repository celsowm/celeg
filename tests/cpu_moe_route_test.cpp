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
}
