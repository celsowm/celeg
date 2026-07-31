#include "celeg/model/config/config.hpp"
#include "support/assertions.hpp"
#include "celeg/model/config/shape.hpp"
#include "celeg/model/config/variant.hpp"
#include <iostream>
#include <string>

namespace {

celeg::ModelShape make_shape(int hidden, int intermediate, int layers,
                            int q_heads, int kv_heads, int head_dim,
                            int vocab, int conv_cache, int conv_dim,
                            int attention_layers, int conv_layers) {
    celeg::ModelShape shape;
    shape.hidden = hidden;
    shape.intermediate = intermediate;
    shape.num_hidden_layers = layers;
    shape.num_attention_heads = q_heads;
    shape.num_key_value_heads = kv_heads;
    shape.head_dim = head_dim;
    shape.vocab_size = vocab;
    shape.conv_cache = conv_cache;
    shape.conv_dim = conv_dim;
    shape.max_position_embeddings = 128000;
    shape.bos_token_id = 1;
    shape.eos_token_id = 7;
    shape.pad_token_id = 0;
    shape.norm_eps = 1e-5f;
    shape.rope_theta = 1'000'000.0f;
    shape.rope_type = "default";
    shape.layer_types.assign(static_cast<size_t>(layers), celeg::LayerType::Convolution);
    if (attention_layers <= 0) {
        throw std::invalid_argument("attention_layers must be positive");
    }
    const int step = layers / attention_layers;
    for (int i = 0; i < attention_layers; ++i) {
        const int idx = (i + 1) * step - 1;
        if (idx >= 0 && idx < layers) {
            shape.layer_types[static_cast<size_t>(idx)] = celeg::LayerType::FullAttention;
        }
    }
    shape.compute_derived();
    shape.validate();
    if (shape.attention_layer_count != attention_layers ||
        shape.conv_layer_count != conv_layers) {
        throw std::logic_error("test shape distribution mismatch");
    }
    return shape;
}

} // namespace

int main() {
    celeg::register_builtin_variants();
    auto& registry = celeg::ModelVariantRegistry::instance();

    const auto ids = registry.ids();
    CELEG_TEST_CHECK(ids.size() == 6);
    CELEG_TEST_CHECK(registry.find("granite") != nullptr);
    CELEG_TEST_CHECK(registry.find("lfm2.5-8b-a1b") != nullptr);
    CELEG_TEST_CHECK(registry.find("lfm2-8b-a1b") != nullptr);

    const celeg::ModelShape shape_230m = make_shape(
        1024, 2560, 14, 16, 8, 64, 65536, 3, 1024, 6, 8);
    const celeg::IModelVariant& variant_230m = registry.select(shape_230m);
    CELEG_TEST_CHECK(variant_230m.id() == "lfm2.5-230m");
    CELEG_TEST_CHECK(variant_230m.repo_id() == "LiquidAI/LFM2.5-230M");
    CELEG_TEST_CHECK(variant_230m.chat_template_kind() == celeg::ChatTemplateKind::Lfm2Instruct);
    CELEG_TEST_CHECK(variant_230m.label().find("230M") != std::string::npos);

    const celeg::ModelShape shape_1_2b = make_shape(
        2048, 12288, 16, 32, 8, 64, 65536, 3, 2048, 6, 10);
    const celeg::IModelVariant& variant_1_2b = registry.select(shape_1_2b);
    CELEG_TEST_CHECK(variant_1_2b.id() == "lfm2.5-1.2b-instruct");
    CELEG_TEST_CHECK(variant_1_2b.repo_id() == "LiquidAI/LFM2.5-1.2B-Instruct");
    CELEG_TEST_CHECK(variant_1_2b.chat_template_kind() == celeg::ChatTemplateKind::Lfm2Instruct);
    CELEG_TEST_CHECK(variant_1_2b.label().find("1.2B-Instruct") != std::string::npos);

    // config.json advertises intermediate_size=12288 but the published
    // checkpoint actually stores w1/w3 with 8192 rows. resolve_shape() must
    // canonicalize to the real value so the weight loader expects the right
    // tensor dimensions.
    const celeg::ModelShape resolved_1_2b = variant_1_2b.resolve_shape(shape_1_2b);
    CELEG_TEST_CHECK(resolved_1_2b.intermediate == 8192);

    // A shape built directly from the real checkpoint (intermediate=8192)
    // must also match this variant so loading from a corrected config works.
    const celeg::ModelShape shape_1_2b_real = make_shape(
        2048, 8192, 16, 32, 8, 64, 65536, 3, 2048, 6, 10);
    CELEG_TEST_CHECK(variant_1_2b.matches(shape_1_2b_real));
    const celeg::ModelShape resolved_real = variant_1_2b.resolve_shape(shape_1_2b_real);
    CELEG_TEST_CHECK(resolved_real.intermediate == 8192);

    // LFM2.5-1.2B-Thinking shares the Instruct topology. It must NOT match on
    // shape alone (would create an ambiguous registry), but must match when the
    // repo hint identifies it as the Thinking checkpoint.
    const celeg::ModelShape shape_thinking = make_shape(
        2048, 12288, 16, 32, 8, 64, 65536, 3, 2048, 6, 10);
    bool thinking_matches_shape_only = true;
    try {
        const celeg::IModelVariant& v = registry.select(shape_thinking);
        (void)v;
        thinking_matches_shape_only = (v.id() == "lfm2.5-1.2b-thinking");
    } catch (const std::runtime_error&) {
        thinking_matches_shape_only = false;
    }
    CELEG_TEST_CHECK(!thinking_matches_shape_only);  // never dominant on shape alone

    const celeg::IModelVariant& variant_thinking =
        registry.select(shape_thinking, "LiquidAI/LFM2.5-1.2B-Thinking");
    CELEG_TEST_CHECK(variant_thinking.id() == "lfm2.5-1.2b-thinking");
    CELEG_TEST_CHECK(variant_thinking.repo_id() == "LiquidAI/LFM2.5-1.2B-Thinking");
    CELEG_TEST_CHECK(variant_thinking.chat_template_kind() == celeg::ChatTemplateKind::Lfm2Instruct);
    CELEG_TEST_CHECK(variant_thinking.label().find("1.2B-Thinking") != std::string::npos);
    CELEG_TEST_CHECK(variant_thinking.resolve_shape(shape_thinking).intermediate == 8192);

    // A Thinking repo hint must NOT select the Instruct variant.
    const celeg::IModelVariant& variant_instruct_again =
        registry.select(shape_1_2b, "LiquidAI/LFM2.5-1.2B-Thinking");
    CELEG_TEST_CHECK(variant_instruct_again.id() == "lfm2.5-1.2b-thinking");

    // A Thinking config resolved via its own hint must differ from Instruct.
    CELEG_TEST_CHECK(variant_thinking.id() != variant_1_2b.id());

    CELEG_TEST_CHECK(registry.find("lfm2.5-230m") != nullptr);
    CELEG_TEST_CHECK(registry.find("lfm2.5-1.2b-instruct") != nullptr);
    CELEG_TEST_CHECK(registry.find("lfm2.5-1.2b-thinking") != nullptr);
    CELEG_TEST_CHECK(registry.find("nonexistent-variant") == nullptr);

    // A valid but unrecognized shape must be rejected.
    const celeg::ModelShape shape_unknown = make_shape(
        4096, 4096, 8, 64, 8, 64, 100, 3, 4096, 4, 4);
    bool rejected = false;
    try {
        registry.select(shape_unknown);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);

    // ---- LFM2.5-8B-A1B (MoE) variant ----
    {
        celeg::ModelShape shape_8b;
        shape_8b.architecture = celeg::ArchitectureKind::MoeLfm2;
        shape_8b.hidden = 2048;
        shape_8b.intermediate = 7168;
        shape_8b.dense_intermediate = 7168;
        shape_8b.moe_intermediate = 1792;
        shape_8b.num_hidden_layers = 24;
        shape_8b.num_attention_heads = 32;
        shape_8b.num_key_value_heads = 8;
        shape_8b.head_dim = 64;
        shape_8b.vocab_size = 128000;
        shape_8b.conv_cache = 3;
        shape_8b.conv_dim = 2048;
        shape_8b.max_position_embeddings = 128000;
        shape_8b.bos_token_id = 124894;
        shape_8b.eos_token_id = 124900;
        shape_8b.pad_token_id = 124893;
        shape_8b.norm_eps = 1e-5f;
        shape_8b.rope_theta = 5'000'000.0f;
        shape_8b.rope_type = "default";
        shape_8b.num_dense_layers = 2;
        shape_8b.num_experts = 32;
        shape_8b.experts_per_token = 4;
        shape_8b.normalize_topk = true;
        shape_8b.use_expert_bias = false;
        shape_8b.routed_scaling_factor = 1.0f;
        // layer_types: 6 full_attention + 18 conv (matching the official
        // checkpoint ordering).
        const std::vector<int> attn_pos = {2, 6, 10, 14, 18, 21};
        shape_8b.layer_types.assign(24, celeg::LayerType::Convolution);
        for (int p : attn_pos) {
            shape_8b.layer_types[static_cast<size_t>(p)] = celeg::LayerType::FullAttention;
        }
        shape_8b.compute_derived();
        shape_8b.validate();

        const celeg::IModelVariant& variant_8b = registry.select(shape_8b);
        CELEG_TEST_CHECK(variant_8b.id() == "lfm2.5-8b-a1b");
        CELEG_TEST_CHECK(variant_8b.repo_id() == "LiquidAI/LFM2.5-8B-A1B");
        CELEG_TEST_CHECK(variant_8b.chat_template_kind() == celeg::ChatTemplateKind::Lfm2Instruct);
        CELEG_TEST_CHECK(variant_8b.label().find("8B-A1B") != std::string::npos);
        CELEG_TEST_CHECK(variant_8b.resolve_shape(shape_8b).intermediate == 7168);

        // A dense 1.2B shape must NOT select the MoE variant.
        bool selected_moe = false;
        try {
            const celeg::IModelVariant& v = registry.select(shape_1_2b);
            selected_moe = (v.id() == "lfm2.5-8b-a1b");
        } catch (const std::runtime_error&) {
            selected_moe = false;
        }
        CELEG_TEST_CHECK(!selected_moe);

        // The repo hint must select the 8B variant even though the base match
        // already succeeds on shape alone.
        const celeg::IModelVariant& variant_8b_hint =
            registry.select(shape_8b, "LiquidAI/LFM2.5-8B-A1B");
        CELEG_TEST_CHECK(variant_8b_hint.id() == "lfm2.5-8b-a1b");

        // LFM2-8B-A1B shares the MoE topology but uses vocab_size 65536 and a
        // different rope_theta, so it must NOT be selected by the LFM2.5 shape.
        celeg::ModelShape shape_lfm2 = shape_8b;
        shape_lfm2.vocab_size = 65536;
        shape_lfm2.rope_theta = 1'000'000.0f;
        shape_lfm2.num_dense_layers = 2;
        shape_lfm2.use_expert_bias = true;
        shape_lfm2.compute_derived();
        shape_lfm2.validate();
        CELEG_TEST_CHECK(!registry.select(shape_lfm2).matches(shape_8b));
        const celeg::IModelVariant& variant_lfm2 =
            registry.select(shape_lfm2, "LiquidAI/LFM2-8B-A1B");
        CELEG_TEST_CHECK(variant_lfm2.id() == "lfm2-8b-a1b");
        CELEG_TEST_CHECK(variant_lfm2.repo_id() == "LiquidAI/LFM2-8B-A1B");
        CELEG_TEST_CHECK(variant_lfm2.chat_template_kind() == celeg::ChatTemplateKind::Lfm2Instruct);
        CELEG_TEST_CHECK(variant_lfm2.label().find("LFM2-8B-A1B") != std::string::npos);
        CELEG_TEST_CHECK(variant_lfm2.resolve_shape(shape_lfm2).intermediate == 7168);
    }

    std::cout << "model_variant_test: ok variants=";
    for (const auto id : registry.ids()) {
        std::cout << id << ' ';
    }
    std::cout << '\n';
}
