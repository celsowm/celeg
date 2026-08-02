#include "celeg/backend/cpu/sampler.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <vector>

int main() {
    celeg::RuntimeTopology shape;
    shape.vocab_size = 4;

    celeg::GenerationConfig greedy;
    greedy.temperature = 0.0f;
    const std::vector<float> logits{1.0f, 4.0f, 2.0f, 3.0f};
    std::vector<std::uint8_t> seen(4, 0);
    std::uint64_t rng = 17;
    CELEG_TEST_CHECK(celeg::CpuSampler::sample(logits, shape, greedy, seen, rng) == 1);

    celeg::GenerationConfig sampled;
    sampled.temperature = 1.0f;
    sampled.top_k = 2;
    sampled.top_p = 1.0f;
    sampled.seed = 123;
    rng = sampled.seed;
    const auto first = celeg::CpuSampler::sample(logits, shape, sampled, seen, rng);
    rng = sampled.seed;
    const auto second = celeg::CpuSampler::sample(logits, shape, sampled, seen, rng);
    CELEG_TEST_CHECK(first == second);
    CELEG_TEST_CHECK(first == 1 || first == 3);
}
