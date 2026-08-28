#include "../canonical_internal.hpp"
#include "../rules.hpp"
#include "../support.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"

namespace celeg::inference_detail {

void bind_dense_ffn(CanonicalInferenceContext& context,
                    int layer,
                    int layer_intermediate) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& bindings = context.facts.bindings;

    const int physical_layer = context.physical_layer(layer);
    const auto* gate = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(physical_layer, "w_gate.weight"),
        TensorRole::FfnGate,
        layer,
        {layer_intermediate, *m.core.hidden_size},
        {});
    add_binding(bindings, TensorRole::FfnGate, layer, *gate, {});

    const auto* up = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(physical_layer, "w_up.weight"),
        TensorRole::FfnUp,
        layer,
        {layer_intermediate, *m.core.hidden_size},
        {});
    add_binding(bindings, TensorRole::FfnUp, layer, *up, {});

    const auto* down = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(physical_layer, "w_down.weight"),
        TensorRole::FfnDown,
        layer,
        {*m.core.hidden_size, layer_intermediate},
        {});
    add_binding(bindings, TensorRole::FfnDown, layer, *down, {});
}

void bind_moe(CanonicalInferenceContext& context,
              int layer,
              std::string_view index) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& facts = context.facts;
    const std::string prefix =
        "model.layers." + std::string(index) + ".mlp.";
    const std::string feed_forward_prefix =
        "model.layers." + std::string(index) + ".feed_forward.";
    const auto has_tensor = [&](std::string_view name) {
        return input.inventory.find(name) != nullptr;
    };

    const auto bind = [&](TensorRole role,
                          std::string name,
                          std::initializer_list<std::int64_t> shape,
                          int expert = -1) {
        const auto* tensor = find_unique(
            input.inventory,
            {std::move(name)},
            role,
            layer,
            shape,
            {});
        facts.bindings.values.push_back({
            role,
            layer,
            expert,
            -1,
            tensor->name,
            tensor->shape,
            {},
        });
    };

    const auto bind_packed = [&](TensorRole role, const std::string& name) {
        const auto* tensor = input.inventory.find(name);
        if (tensor == nullptr) {
            fail(ResolutionFailureKind::MissingTensorRole,
                 "automatic resolution could not bind " +
                     std::string(tensor_role_name(role)) + " for layer " +
                     std::to_string(layer));
        }
        facts.bindings.values.push_back({
            role, layer, -1, -1, tensor->name, tensor->shape, {},
        });
    };

    const int num_experts = context.moe->num_experts;
    const std::string router_name = has_tensor(prefix + "gate.weight")
        ? prefix + "gate.weight"
        : feed_forward_prefix + "gate.weight";
    bind(
        TensorRole::MoeRouter,
        router_name,
        {num_experts, *m.core.hidden_size});
    std::optional<std::string> bias_name;
    for (const std::string& candidate : {
             prefix + "gate.expert_bias",
             feed_forward_prefix + "expert_bias.weight",
             feed_forward_prefix + "expert_bias",
         }) {
        if (has_tensor(candidate)) {
            bias_name = candidate;
            break;
        }
    }
    if (bias_name) {
        bind(
            TensorRole::MoeRouterBias,
            *bias_name,
            {num_experts});
    }

    const auto* moe = std::get_if<MixtureOfExpertsSpec>(
        &facts.graph.layers[static_cast<size_t>(layer)].feed_forward);
    if (moe == nullptr || moe->intermediate_size <= 0) {
        fail(
            ResolutionFailureKind::UnsupportedGraphPrimitive,
            "MoE binding resolution has no per-layer routed width: " +
                std::to_string(layer));
    }

    const int expert_intermediate = moe->intermediate_size;
    const bool named_individual = has_tensor(prefix + "experts.0.gate_proj.weight");
    const bool raw_individual = !named_individual &&
        has_tensor(feed_forward_prefix + "experts.0.w1.weight");
    std::optional<std::string> packed_prefix;
    if (!named_individual && !raw_individual) {
        const std::string alternate_prefix =
            "model.language_model.layers." + std::string(index) + ".mlp.";
        for (const std::string& candidate_prefix : {prefix, alternate_prefix}) {
            if (has_tensor(candidate_prefix + "experts.gate_up_proj") &&
                has_tensor(candidate_prefix + "experts.down_proj")) {
                packed_prefix = candidate_prefix;
                break;
            }
        }
    }

    if (named_individual) {
        for (int expert = 0; expert < num_experts; ++expert) {
            const std::string expert_prefix =
                prefix + "experts." + std::to_string(expert) + ".";
            bind(
                TensorRole::MoeExpertGate,
                expert_prefix + "gate_proj.weight",
                {expert_intermediate, *m.core.hidden_size},
                expert);
            bind(
                TensorRole::MoeExpertUp,
                expert_prefix + "up_proj.weight",
                {expert_intermediate, *m.core.hidden_size},
                expert);
            bind(
                TensorRole::MoeExpertDown,
                expert_prefix + "down_proj.weight",
                {*m.core.hidden_size, expert_intermediate},
                expert);
        }
    } else if (raw_individual) {
        for (int expert = 0; expert < num_experts; ++expert) {
            const std::string expert_prefix =
                feed_forward_prefix + "experts." + std::to_string(expert) + ".";
            bind(
                TensorRole::MoeExpertGate,
                expert_prefix + "w1.weight",
                {expert_intermediate, *m.core.hidden_size},
                expert);
            bind(
                TensorRole::MoeExpertUp,
                expert_prefix + "w3.weight",
                {expert_intermediate, *m.core.hidden_size},
                expert);
            bind(
                TensorRole::MoeExpertDown,
                expert_prefix + "w2.weight",
                {*m.core.hidden_size, expert_intermediate},
                expert);
        }
    } else if (packed_prefix) {
        bind_packed(TensorRole::MoePackedGateUp, *packed_prefix + "experts.gate_up_proj");
        bind_packed(TensorRole::MoePackedDown, *packed_prefix + "experts.down_proj");
    } else {
        fail(
            ResolutionFailureKind::MissingTensorRole,
            "automatic resolution could not determine the MoE routed-expert "
            "checkpoint layout for layer " + std::to_string(layer));
    }

    if (moe->shared) {
        const int shared_intermediate = moe->shared->intermediate_size;
        const std::string shared = prefix + "shared_experts.";
        bind(
            TensorRole::MoeSharedGate,
            shared + "gate_proj.weight",
            {shared_intermediate, *m.core.hidden_size});
        bind(
            TensorRole::MoeSharedUp,
            shared + "up_proj.weight",
            {shared_intermediate, *m.core.hidden_size});
        bind(
            TensorRole::MoeSharedDown,
            shared + "down_proj.weight",
            {*m.core.hidden_size, shared_intermediate});
    }
}

void resolve_layer_feed_forward(CanonicalInferenceContext& context,
                                int layer) {
    auto& facts = context.facts;
    const LayerSpec& semantic_layer =
        facts.graph.layers[static_cast<size_t>(layer)];

    if (std::holds_alternative<MlpBlockSpec>(semantic_layer.mixer)) return;
    if (std::holds_alternative<std::monostate>(semantic_layer.feed_forward)) return;

    if (std::holds_alternative<MixtureOfExpertsSpec>(semantic_layer.feed_forward)) {
        bind_moe(context, layer, std::to_string(context.physical_layer(layer)));
    } else {
        bind_dense_ffn(context, layer,
                       context.intermediate_sizes.at(static_cast<size_t>(layer)));
    }
}

}
