#include "inference/rules.hpp"
#include "inference/support.hpp"

#include "celeg/model/inference.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace idi = celeg::inference_detail;

celeg::TensorInventory inventory_of(
    std::vector<celeg::TensorInventoryEntry> entries) {
    return celeg::TensorInventory(std::move(entries));
}

celeg::TensorInventoryEntry tensor(const std::string& name,
                                   std::vector<int64_t> shape) {
    return celeg::TensorInventoryEntry{
        name, std::move(shape), celeg::TensorDType::BF16};
}

std::unique_ptr<idi::CanonicalInferenceContext> make_context(
    const celeg::InferenceInput& input) {
    auto context = std::make_unique<idi::CanonicalInferenceContext>(input);
    context->layer_count = 1;
    context->facts.graph.layers.resize(1);
    context->intermediate_sizes = {8};
    return context;
}

/// A synthetic grammar for an attention family spelled in a convention no
/// built-in rule knows. Registering it into the catalog is the entire edit
/// surface for recognizing the new spelling: no cascade or binding dispatch
/// anywhere else changes.
class CustomAttentionSpellingRule final : public idi::ILayerInferenceRule {
public:
    explicit CustomAttentionSpellingRule(int specificity)
        : specificity_(specificity) {}

    std::string_view id() const override {
        return "custom_attention_spelling";
    }
    int specificity() const override { return specificity_; }
    idi::MixerFamily family() const override { return idi::MixerFamily::Attention; }

    bool probe(const idi::CanonicalInferenceContext& context, int layer)
        const override {
        return context.input.inventory.find(
                   "custom.blocks." + std::to_string(layer) +
                   ".attn.q_weight") != nullptr;
    }

    void resolve(idi::CanonicalInferenceContext& context, int layer)
        const override {
        const auto& inventory = context.input.inventory;
        celeg::LayerSpec& semantic_layer =
            context.facts.graph.layers[static_cast<size_t>(layer)];
        const struct {
            celeg::TensorRole role;
            const char* suffix;
        } parts[] = {
            {celeg::TensorRole::AttentionQuery, "q_weight"},
            {celeg::TensorRole::AttentionKey, "k_weight"},
            {celeg::TensorRole::AttentionValue, "v_weight"},
            {celeg::TensorRole::AttentionOutput, "o_weight"},
        };
        celeg::AttentionSpec attention;
        attention.query_heads = 1;
        attention.key_value_heads = 1;
        attention.head_dim = 8;
        attention.pattern = celeg::FullCausalPattern{};
        attention.query_scale = 1.0f;
        for (const auto& part : parts) {
            const auto* entry = inventory.find(
                "custom.blocks." + std::to_string(layer) + ".attn." +
                part.suffix);
            if (entry == nullptr || entry->shape != std::vector<int64_t>{8, 8}) {
                idi::fail(celeg::ResolutionFailureKind::ShapeConstraintViolation,
                     "custom grammar tensor is missing or misshaped: " +
                         std::string(part.suffix));
            }
            idi::add_binding(context.facts.bindings, part.role, layer, *entry, {});
        }
        semantic_layer.mixer = std::move(attention);
        if (!idi::layer_has_feed_forward(context, layer)) {
            semantic_layer.feed_forward = std::monostate{};
        }
    }

private:
    int specificity_;
};

void run_registration_extends_recognition() {
    celeg::InferenceInput input;
    input.inventory = inventory_of({
        tensor("custom.blocks.0.attn.q_weight", {8, 8}),
        tensor("custom.blocks.0.attn.k_weight", {8, 8}),
        tensor("custom.blocks.0.attn.v_weight", {8, 8}),
        tensor("custom.blocks.0.attn.o_weight", {8, 8}),
    });
    auto context = make_context(input);

    auto rules = idi::make_builtin_layer_inference_rules();
    rules.push_back(std::make_unique<CustomAttentionSpellingRule>(2));

    const idi::ILayerInferenceRule& selected =
        idi::select_layer_inference_rule(rules, *context, 0);
    CELEG_TEST_CHECK(selected.id() == "custom_attention_spelling");

    selected.resolve(*context, 0);
    const celeg::LayerSpec& layer = context->facts.graph.layers.front();
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(layer.mixer));
    CELEG_TEST_CHECK(
        std::holds_alternative<std::monostate>(layer.feed_forward));
    const auto* query = context->facts.bindings.find(
        celeg::TensorRole::AttentionQuery, 0);
    CELEG_TEST_CHECK(query != nullptr);
    CELEG_TEST_CHECK(query->source_name == "custom.blocks.0.attn.q_weight");
    const auto* output = context->facts.bindings.find(
        celeg::TensorRole::AttentionOutput, 0);
    CELEG_TEST_CHECK(output != nullptr);
    CELEG_TEST_CHECK(output->source_name == "custom.blocks.0.attn.o_weight");
}

void run_equal_specificity_is_a_conflict() {
    celeg::InferenceInput input;
    input.inventory = inventory_of({
        tensor("custom.blocks.0.attn.q_weight", {8, 8}),
        tensor("model.layers.0.self_attn.q_proj.weight", {8, 8}),
    });
    auto context = make_context(input);

    auto rules = idi::make_builtin_layer_inference_rules();
    rules.push_back(std::make_unique<CustomAttentionSpellingRule>(2));

    bool conflicted = false;
    try {
        idi::select_layer_inference_rule(rules, *context, 0);
    } catch (const celeg::ResolutionError& error) {
        conflicted =
            error.kind() == celeg::ResolutionFailureKind::ConflictingInferenceFacts;
    }
    CELEG_TEST_CHECK(conflicted);
}

void run_attention_recurrent_exclusion() {
    celeg::InferenceInput input;
    input.inventory = inventory_of({
        tensor("model.layers.0.self_attn.q_proj.weight", {8, 8}),
        tensor("model.layers.0.mixer.in_proj.weight", {8, 8}),
    });
    auto context = make_context(input);

    const auto rules = idi::make_builtin_layer_inference_rules();
    bool conflicted = false;
    try {
        idi::select_layer_inference_rule(rules, *context, 0);
    } catch (const celeg::ResolutionError& error) {
        conflicted =
            error.kind() == celeg::ResolutionFailureKind::ConflictingInferenceFacts;
    }
    CELEG_TEST_CHECK(conflicted);
}

void run_specificity_decides_between_grammars() {
    celeg::InferenceInput input;
    input.inventory = inventory_of({
        tensor("custom.blocks.0.attn.q_weight", {8, 8}),
        tensor("model.layers.0.self_attn.q_proj.weight", {8, 8}),
    });
    auto context = make_context(input);

    auto rules = idi::make_builtin_layer_inference_rules();
    rules.push_back(std::make_unique<CustomAttentionSpellingRule>(1));

    const idi::ILayerInferenceRule& selected =
        idi::select_layer_inference_rule(rules, *context, 0);
    CELEG_TEST_CHECK(selected.id() == "standard_attention");
}

void run_builtin_catalog_specificities_are_distinct() {
    const auto rules = idi::make_builtin_layer_inference_rules();
    CELEG_TEST_CHECK(rules.size() == 7);
    for (size_t i = 0; i < rules.size(); ++i) {
        for (size_t j = i + 1; j < rules.size(); ++j) {
            CELEG_TEST_CHECK(rules[i]->specificity() != rules[j]->specificity());
            CELEG_TEST_CHECK(rules[i]->id() != rules[j]->id());
        }
    }
}

}

int main() {
    run_builtin_catalog_specificities_are_distinct();
    run_registration_extends_recognition();
    run_equal_specificity_is_a_conflict();
    run_attention_recurrent_exclusion();
    run_specificity_decides_between_grammars();
    std::cout << "layer_inference_rule_test: ok\n";
    return 0;
}
