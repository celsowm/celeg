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

// A checkpoint whose MoE tensors use deliberately unconventional spellings
// that no backend's (former) literal-substring sniffing would recognize.
// The naming policy is the only thing that knows how to find them; the
// weight plan must still resolve the correct role/locator for each,
// proving that layout decisions (packed vs individual) and per-tensor
// binding both flow from the resolved plan rather than from backends
// re-matching tensor-name spellings themselves.
class PoisonedMoeRepository final : public celeg::IWeightRepository {
public:
    bool contains(std::string_view name) const override {
        return name == "poison.layer0.expert0.gate" ||
               name == "poison.layer0.expert0.up" ||
               name == "poison.layer0.expert0.down" ||
               name == "poison.layer0.router_bias" ||
               name == "poison.layer1.packed_gate_up" ||
               name == "poison.layer1.packed_down" ||
               name == "poison.layer1.router_bias";
    }

    celeg::HostTensorView tensor(std::string_view name) const override {
        if (!contains(name)) return {};
        return celeg::HostTensorView{.dtype = celeg::TensorDType::BF16,
                                     .shape = {}, .data = nullptr, .bytes = 0};
    }

    std::vector<std::string> names() const override {
        return {"poison.layer0.expert0.gate", "poison.layer0.expert0.up",
                "poison.layer0.expert0.down", "poison.layer0.router_bias",
                "poison.layer1.packed_gate_up", "poison.layer1.packed_down",
                "poison.layer1.router_bias"};
    }
};

class PoisonedMoeNamingPolicy final : public celeg::ITensorNamingPolicy {
public:
    std::vector<std::string> candidates(
        const celeg::TensorRequest& request) const override {
        // Deliberately offer decoy candidates first (conventional spellings
        // that do NOT exist in the repository) to prove selection is driven
        // by existence, not by pattern order or literal recognition.
        if (request.role == celeg::TensorRole::MoeExpertGate &&
            request.layer == 0 && request.expert == 0) {
            return {"model.layers.0.mlp.experts.0.gate_proj.weight",
                    "poison.layer0.expert0.gate"};
        }
        if (request.role == celeg::TensorRole::MoeExpertUp &&
            request.layer == 0 && request.expert == 0) {
            return {"model.layers.0.mlp.experts.0.up_proj.weight",
                    "poison.layer0.expert0.up"};
        }
        if (request.role == celeg::TensorRole::MoeExpertDown &&
            request.layer == 0 && request.expert == 0) {
            return {"model.layers.0.mlp.experts.0.down_proj.weight",
                    "poison.layer0.expert0.down"};
        }
        if (request.role == celeg::TensorRole::MoePackedGateUp &&
            request.layer == 1) {
            return {"model.layers.1.mlp.experts.gate_up_proj",
                    "poison.layer1.packed_gate_up"};
        }
        if (request.role == celeg::TensorRole::MoePackedDown &&
            request.layer == 1) {
            return {"model.layers.1.mlp.experts.down_proj",
                    "poison.layer1.packed_down"};
        }
        if (request.role == celeg::TensorRole::MoeRouterBias &&
            request.layer == 0) {
            return {"model.layers.0.mlp.gate.expert_bias",
                    "poison.layer0.router_bias"};
        }
        if (request.role == celeg::TensorRole::MoeRouterBias &&
            request.layer == 1) {
            return {"model.layers.1.feed_forward.expert_bias.weight",
                    "poison.layer1.router_bias"};
        }
        return {};
    }
};

celeg::LayerSpec make_moe_layer(int intermediate, int num_experts, int hidden) {
    celeg::LayerSpec layer;
    layer.mixer_norm.before = celeg::NormSpec{1.0e-5f, celeg::NormWeightKind::Scale};
    layer.feed_forward_norm.before = celeg::NormSpec{1.0e-5f, celeg::NormWeightKind::Scale};
    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = hidden;
    attention.position = celeg::RopePositionSpec{10000.0, 1.0, {}};
    layer.mixer = attention;
    celeg::MixtureOfExpertsSpec moe;
    moe.intermediate_size = intermediate;
    moe.num_experts = num_experts;
    moe.experts_per_token = 1;
    moe.use_expert_bias = true;
    layer.feed_forward = moe;
    return layer;
}

void run_poisoned_moe_layout_test() {
    auto repository = std::make_shared<PoisonedMoeRepository>();
    celeg::ResolvedModel model;
    model.provenance.identity = "poisoned-moe-boundary";
    model.graph.hidden = 4;
    model.graph.final_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
    model.graph.layers.push_back(make_moe_layer(2, 1, 4));
    model.graph.layers.push_back(make_moe_layer(2, 1, 4));

    PoisonedMoeNamingPolicy naming;
    celeg::build_weight_plan_from_graph(model, naming, repository.get());

    const auto& requests = model.weight_plan.requests;
    const auto find_one = [&](celeg::TensorRole role, int layer, int expert) {
        for (const auto& request : requests) {
            if (request.role == role && request.layer == layer &&
                request.expert == expert) {
                return &request;
            }
        }
        return static_cast<const celeg::TensorRequest*>(nullptr);
    };

    // Layer 0: individual experts, resolved to the poisoned spelling even
    // though a conventional-looking decoy candidate was offered first.
    const auto* gate0 = find_one(celeg::TensorRole::MoeExpertGate, 0, 0);
    CELEG_TEST_CHECK(gate0 != nullptr);
    CELEG_TEST_CHECK(gate0->source_name.has_value());
    CELEG_TEST_CHECK(*gate0->source_name == "poison.layer0.expert0.gate");

    // Layer 1: no individual per-expert request should have been planned at
    // all -- the plan itself decided this layer is packed (using the same
    // existence-checked naming-policy candidates), so a backend consuming
    // this plan never needs to sniff repo.contains() on literal spellings
    // to find that out.
    CELEG_TEST_CHECK(find_one(celeg::TensorRole::MoeExpertGate, 1, 0) == nullptr);
    const auto* packed_gate_up = find_one(celeg::TensorRole::MoePackedGateUp, 1, -1);
    CELEG_TEST_CHECK(packed_gate_up != nullptr);
    CELEG_TEST_CHECK(packed_gate_up->source_name.has_value());
    CELEG_TEST_CHECK(*packed_gate_up->source_name == "poison.layer1.packed_gate_up");
    const auto* packed_down = find_one(celeg::TensorRole::MoePackedDown, 1, -1);
    CELEG_TEST_CHECK(packed_down != nullptr);
    CELEG_TEST_CHECK(packed_down->source_name.has_value());
    CELEG_TEST_CHECK(*packed_down->source_name == "poison.layer1.packed_down");

    // The router bias is planned whenever the MoE semantics say one exists,
    // so backends read its resolved name instead of probing spellings.
    const auto* bias0 = find_one(celeg::TensorRole::MoeRouterBias, 0, -1);
    CELEG_TEST_CHECK(bias0 != nullptr);
    CELEG_TEST_CHECK(bias0->source_name.has_value());
    CELEG_TEST_CHECK(*bias0->source_name == "poison.layer0.router_bias");
    const auto* bias1 = find_one(celeg::TensorRole::MoeRouterBias, 1, -1);
    CELEG_TEST_CHECK(bias1 != nullptr);
    CELEG_TEST_CHECK(bias1->source_name.has_value());
    CELEG_TEST_CHECK(*bias1->source_name == "poison.layer1.router_bias");

    // The per-layer resolved-name bundle backends consume must carry the
    // same poisoned spellings and the same layout decision as the plan.
    const celeg::MoeExpertTensorNames individual_names =
        celeg::moe_expert_tensor_names(requests, 0, 1);
    CELEG_TEST_CHECK(!individual_names.packed());
    CELEG_TEST_CHECK(individual_names.gate.size() == 1);
    CELEG_TEST_CHECK(individual_names.up.size() == 1);
    CELEG_TEST_CHECK(individual_names.down.size() == 1);
    CELEG_TEST_CHECK(individual_names.gate.front() == "poison.layer0.expert0.gate");
    CELEG_TEST_CHECK(individual_names.up.front() == "poison.layer0.expert0.up");
    CELEG_TEST_CHECK(individual_names.down.front() == "poison.layer0.expert0.down");

    const celeg::MoeExpertTensorNames packed_names =
        celeg::moe_expert_tensor_names(requests, 1, 1);
    CELEG_TEST_CHECK(packed_names.packed());
    CELEG_TEST_CHECK(packed_names.packed_gate_up == "poison.layer1.packed_gate_up");
    CELEG_TEST_CHECK(packed_names.packed_down == "poison.layer1.packed_down");
}

}

int main() {
    auto repository = std::make_shared<FakeRepository>();
    celeg::CheckpointView checkpoint;
    checkpoint.repository = repository;
    CELEG_TEST_CHECK(checkpoint.repository->contains("fake.attention_input_norm.0"));

    celeg::ResolvedModel model;
    model.provenance.identity = "fake-repository-boundary";
    celeg::LayerSpec layer;
    model.graph.hidden = 4;
    model.graph.final_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = 4;
    attention.position = celeg::RopePositionSpec{10000.0, 1.0, {}};
    layer.mixer_norm.before = celeg::NormSpec{1.0e-5f, celeg::NormWeightKind::Scale};
    layer.feed_forward_norm.before = celeg::NormSpec{1.0e-5f, celeg::NormWeightKind::Scale};
    layer.mixer = attention;
    layer.feed_forward = celeg::DenseFeedForwardSpec{8, celeg::ActivationKind::SwiGLU};
    model.graph.layers.push_back(layer);
    model.weight_plan.requests.push_back({
        celeg::TensorRole::AttentionInputNorm, 0, -1, {4}});
    FakeNamingPolicy naming;
    celeg::resolve_weight_plan(model, naming, repository.get());
    CELEG_TEST_CHECK(model.weight_plan.requests.front().source_name.has_value());
    CELEG_TEST_CHECK(repository->contains(
        *model.weight_plan.requests.front().source_name));

    const auto cpu = celeg::CpuModelCompiler{}.compile(model);
    const auto cuda = celeg::CudaModelCompiler{}.compile(model);
    CELEG_TEST_CHECK(cpu.identity == cuda.identity);
    CELEG_TEST_CHECK(cpu.weight_request_count == 1);
    CELEG_TEST_CHECK(cuda.weight_request_count == 1);

    run_poisoned_moe_layout_test();
    return 0;
}
