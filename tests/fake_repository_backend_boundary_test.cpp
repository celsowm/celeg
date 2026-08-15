#include "celeg/backend/cpu/compiler.hpp"
#include "celeg/backend/cuda/compiler.hpp"
#include "celeg/checkpoint/view.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/weight_plan.hpp"
#include "support/assertions.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

class FakeRepository final : public celeg::IWeightRepository {
public:
    bool contains(std::string_view name) const override {
        return name == "fake.attention_input_norm.0";
    }

    celeg::HostTensorView tensor(std::string_view name) const override {
        if (!contains(name)) return {};
        return celeg::HostTensorView{.dtype = celeg::TensorDType::BF16,
                                     .shape = {4}, .data = nullptr, .bytes = 0};
    }

    std::vector<std::string> names() const override {
        return {"fake.attention_input_norm.0"};
    }
};

class FakeNamingPolicy final : public celeg::ITensorNamingPolicy {
public:
    std::vector<std::string> candidates(
        const celeg::TensorRequest& request) const override {
        if (request.role == celeg::TensorRole::AttentionInputNorm &&
            request.layer == 0) {
            return {"fake.attention_input_norm.0"};
        }
        return {};
    }
};

}

int main() {
    auto repository = std::make_shared<FakeRepository>();
    celeg::CheckpointView checkpoint;
    checkpoint.repository = repository;
    CELEG_TEST_CHECK(checkpoint.repository->contains("fake.attention_input_norm.0"));

    celeg::ResolvedModel model;
    model.provenance.identity = "fake-repository-boundary";
    model.capabilities.supports_cpu = true;
    model.capabilities.supports_cuda = true;
    celeg::LayerSpec layer;
    model.graph.hidden = 4;
    model.graph.final_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = 4;
    attention.position = celeg::RopePositionSpec{10000.0, 1.0, {}};
    layer.operator_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
    layer.mixer = attention;
    layer.feed_forward = celeg::DenseFeedForwardSpec{8, celeg::ActivationKind::SwiGLU};
    model.graph.layers.push_back(layer);
    model.weight_plan.requests.push_back({
        celeg::TensorRole::AttentionInputNorm, 0, -1, {4}});
    FakeNamingPolicy naming;
    celeg::resolve_weight_plan(model, naming);
    CELEG_TEST_CHECK(model.weight_plan.requests.front().source_name.has_value());
    CELEG_TEST_CHECK(repository->contains(
        *model.weight_plan.requests.front().source_name));

    const auto cpu = celeg::CpuModelCompiler{}.compile(model);
    const auto cuda = celeg::CudaModelCompiler{}.compile(model);
    CELEG_TEST_CHECK(cpu.identity == cuda.identity);
    CELEG_TEST_CHECK(cpu.weight_request_count == 1);
    CELEG_TEST_CHECK(cuda.weight_request_count == 1);
    return 0;
}
