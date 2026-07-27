#include "lfm/model/config/config.hpp"
#include "lfm/model/config/shape.hpp"
#include "lfm/model/config/variant.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

void write_230m_config(const std::filesystem::path& path) {
    std::ofstream out(path);
    out << R"({
      "model_type":"lfm2", "dtype":"bfloat16",
      "hidden_size":1024, "intermediate_size":2560,
      "num_hidden_layers":14, "num_attention_heads":16,
      "num_key_value_heads":8, "vocab_size":65536,
      "conv_L_cache":3, "conv_dim":1024,
      "max_position_embeddings":128000,
      "bos_token_id":1, "eos_token_id":7, "pad_token_id":0,
      "norm_eps":1e-5, "conv_bias":false,
      "tie_word_embeddings":true, "use_pos_enc":true,
      "rope_parameters":{"rope_theta":1000000.0,"rope_type":"default"},
      "layer_types":["conv","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv"]
    })";
}

void write_1_2b_config(const std::filesystem::path& path) {
    // Mirrors LiquidAI/LFM2.5-1.2B-Instruct config.json layout: top-level
    // rope_theta, "tie_embedding" alias, "num_heads" alias.
    std::ofstream out(path);
    out << R"({
      "model_type":"lfm2", "dtype":"bfloat16",
      "hidden_size":2048, "intermediate_size":12288,
      "num_hidden_layers":16, "num_heads":32,
      "num_key_value_heads":8, "vocab_size":65536,
      "conv_L_cache":3, "conv_dim":2048,
      "max_position_embeddings":128000,
      "bos_token_id":1, "eos_token_id":7, "pad_token_id":0,
      "norm_eps":1e-5, "conv_bias":false,
      "tie_embedding":true, "use_pos_enc":true,
      "rope_theta":1000000.0,
      "layer_types":["conv","conv","full_attention","conv","conv","full_attention","conv","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv"]
    })";
}

void write_8b_moe_config(const std::filesystem::path& path) {
    // Mirrors the official LiquidAI/LFM2.5-8B-A1B config.json. conv_dim and
    // use_pos_enc are intentionally omitted to exercise the architecture-aware
    // defaults.
    std::ofstream out(path);
    out << R"({
      "_name":"LiquidAI/LFM2.5-8B-A1B",
      "model_type":"lfm2_moe", "dtype":"bfloat16",
      "hidden_size":2048, "intermediate_size":7168,
      "moe_intermediate_size":1792, "num_hidden_layers":24,
      "num_attention_heads":32, "num_key_value_heads":8, "vocab_size":128000,
      "conv_L_cache":3, "max_position_embeddings":128000,
      "bos_token_id":124894, "eos_token_id":124900, "pad_token_id":124893,
      "norm_eps":1e-5, "conv_bias":false, "tie_word_embeddings":true,
      "rope_parameters":{"rope_theta":5000000.0,"rope_type":"default"},
      "num_dense_layers":2, "num_experts":32, "num_experts_per_tok":4,
      "norm_topk_prob":true, "use_expert_bias":true, "routed_scaling_factor":1.0,
      "layer_types":["conv","conv","full_attention","conv","conv","conv","full_attention","conv","conv","conv","full_attention","conv","conv","conv","full_attention","conv","conv","conv","full_attention","conv","conv","full_attention","conv","conv"]
    })";
}

void write_1_2b_thinking_config(const std::filesystem::path& path) {
    // Topologically identical to the Instruct config, but disambiguated by the
    // "_name" field (HuggingFace stores the repo id there).
    std::ofstream out(path);
    out << R"({
      "_name":"LiquidAI/LFM2.5-1.2B-Thinking",
      "model_type":"lfm2", "dtype":"bfloat16",
      "hidden_size":2048, "intermediate_size":12288,
      "num_hidden_layers":16, "num_heads":32,
      "num_key_value_heads":8, "vocab_size":65536,
      "conv_L_cache":3, "conv_dim":2048,
      "max_position_embeddings":32768,
      "bos_token_id":1, "eos_token_id":7, "pad_token_id":0,
      "norm_eps":1e-5, "conv_bias":false,
      "tie_embedding":true, "use_pos_enc":true,
      "rope_theta":1000000.0,
      "layer_types":["conv","conv","full_attention","conv","conv","full_attention","conv","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv"]
    })";
}

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "lfm25-config-test.json";

    write_230m_config(path);
    const lfm::ModelConfig config_230m = lfm::ModelConfig::load(path.string());
    const lfm::ModelShape shape_230m = lfm::ModelShape::from_config(config_230m);
    assert(shape_230m.head_dim == 64);
    assert(shape_230m.num_hidden_layers == 14);
    assert(static_cast<int>(shape_230m.layer_types.size()) == 14);
    assert(shape_230m.attention_layer_count == 6);
    assert(shape_230m.conv_layer_count == 8);
    assert(shape_230m.q_width == 16 * 64);
    assert(shape_230m.kv_width == 8 * 64);
    assert(shape_230m.qkv_width == shape_230m.q_width + 2 * shape_230m.kv_width);
    assert(shape_230m.attention_slot_for_layer[2] == 0);
    assert(shape_230m.attention_slot_for_layer[4] == 1);
    assert(shape_230m.attention_slot_for_layer[0] == -1);
    assert(shape_230m.layer_for_attention_slot.size() == 6);
    assert(shape_230m.layer_for_attention_slot[0] == 2);
    assert(shape_230m.layer_for_attention_slot[5] == 12);
    assert(config_230m.summary().find("attention_layers=6") != std::string::npos);

    lfm::register_builtin_variants();
    const lfm::IModelVariant& variant_230m =
        lfm::ModelVariantRegistry::instance().select(shape_230m);
    assert(variant_230m.id() == "lfm2.5-230m");
    assert(variant_230m.repo_id() == "LiquidAI/LFM2.5-230M");

    write_1_2b_config(path);
    const lfm::ModelConfig config_1_2b = lfm::ModelConfig::load(path.string());
    const lfm::ModelShape shape_1_2b = lfm::ModelShape::from_config(config_1_2b);
    assert(shape_1_2b.head_dim == 64);
    assert(shape_1_2b.num_hidden_layers == 16);
    assert(shape_1_2b.attention_layer_count == 6);
    assert(shape_1_2b.conv_layer_count == 10);
    assert(shape_1_2b.q_width == 32 * 64);
    assert(shape_1_2b.kv_width == 8 * 64);
    assert(config_1_2b.summary().find("attention_layers=6") != std::string::npos);

    const lfm::IModelVariant& variant_1_2b =
        lfm::ModelVariantRegistry::instance().select(shape_1_2b);
    assert(variant_1_2b.id() == "lfm2.5-1.2b-instruct");
    assert(variant_1_2b.repo_id() == "LiquidAI/LFM2.5-1.2B-Instruct");

    // The Thinking checkpoint shares the Instruct topology but must be
    // selected via its "_name" repo hint. Without the hint it must fall back
    // to the Instruct variant (shape-only loading stays unambiguous).
    write_1_2b_thinking_config(path);
    const lfm::ModelConfig config_thinking = lfm::ModelConfig::load(path.string());
    assert(!config_thinking.repo_hint.empty());
    const lfm::ModelShape shape_thinking =
        lfm::ModelShape::from_config(config_thinking);
    const lfm::IModelVariant& variant_fallback =
        lfm::ModelVariantRegistry::instance().select(shape_thinking);
    assert(variant_fallback.id() == "lfm2.5-1.2b-instruct");
    const lfm::IModelVariant& variant_thinking =
        lfm::ModelVariantRegistry::instance().select(shape_thinking,
                                                      config_thinking.repo_hint);
    assert(variant_thinking.id() == "lfm2.5-1.2b-thinking");
    assert(variant_thinking.repo_id() == "LiquidAI/LFM2.5-1.2B-Thinking");
    const lfm::ModelShape resolved_thinking = variant_thinking.resolve_shape(shape_thinking);
    assert(resolved_thinking.intermediate == 8192);

    // ---- LFM2 MoE configuration (LiquidAI/LFM2.5-8B-A1B) ----
    write_8b_moe_config(path);
    const lfm::ModelConfig config_moe = lfm::ModelConfig::load(path.string());
    assert(config_moe.architecture == lfm::ArchitectureKind::MoeLfm2);
    assert(config_moe.moe.has_value());
    assert(config_moe.moe->num_experts == 32);
    assert(config_moe.moe->experts_per_token == 4);
    assert(config_moe.moe->num_dense_layers == 2);
    assert(config_moe.moe->moe_intermediate_size == 1792);
    assert(config_moe.moe->normalize_topk);
    assert(config_moe.moe->use_expert_bias);
    const lfm::ModelShape shape_moe = lfm::ModelShape::from_config(config_moe);
    assert(shape_moe.architecture == lfm::ArchitectureKind::MoeLfm2);
    assert(shape_moe.hidden == 2048);
    assert(shape_moe.dense_intermediate == 7168);
    assert(shape_moe.moe_intermediate == 1792);
    assert(shape_moe.num_dense_layers == 2);
    assert(shape_moe.num_experts == 32);
    assert(shape_moe.experts_per_token == 4);
    assert(shape_moe.attention_layer_count == 6);
    assert(shape_moe.conv_layer_count == 18);
    assert(shape_moe.num_hidden_layers == 24);
    // First two layers are dense; the rest are MoE.
    assert(!shape_moe.layer_uses_moe(0));
    assert(!shape_moe.layer_uses_moe(1));
    assert(shape_moe.layer_uses_moe(2));
    assert(shape_moe.layer_uses_moe(23));
    // Fingerprints must differ between dense and MoE shapes.
    assert(shape_moe.fingerprint() != shape_1_2b.fingerprint());
    assert(shape_moe.fingerprint().find("moe") != std::string::npos);

    // ---- MoE configuration validation ----
    auto expect_config_error = [&path](const std::string& json) {
        std::ofstream out(path);
        out << json;
        bool threw = false;
        try {
            lfm::ModelConfig::load(path.string());
        } catch (const std::runtime_error&) {
            threw = true;
        }
        return threw;
    };
    // experts_per_token exceeds num_experts
    assert(expect_config_error(R"({"model_type":"lfm2_moe","dtype":"bfloat16","hidden_size":64,"intermediate_size":64,"moe_intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"vocab_size":100,"conv_L_cache":3,"max_position_embeddings":1024,"bos_token_id":1,"eos_token_id":2,"pad_token_id":0,"norm_eps":1e-5,"conv_bias":false,"tie_word_embeddings":true,"use_pos_enc":true,"rope_theta":1000000.0,"num_dense_layers":0,"num_experts":4,"num_experts_per_tok":8,"norm_topk_prob":true,"use_expert_bias":true,"layer_types":["conv","conv"]})"));
    // unsupported architecture
    assert(expect_config_error(R"({"model_type":"lfm3","dtype":"bfloat16","hidden_size":64,"intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"vocab_size":100,"conv_L_cache":3,"conv_dim":64,"max_position_embeddings":1024,"bos_token_id":1,"eos_token_id":2,"pad_token_id":0,"norm_eps":1e-5,"conv_bias":false,"tie_word_embeddings":true,"use_pos_enc":true,"rope_theta":1000000.0,"layer_types":["conv","conv"]})"));

    // Unknown shapes must be rejected by the registry.
    lfm::ModelShape bogus;
    bogus.hidden = 4096;
    bogus.intermediate = 4096;
    bogus.num_hidden_layers = 4;
    bogus.num_attention_heads = 4;
    bogus.num_key_value_heads = 4;
    bogus.head_dim = 1024;
    bogus.vocab_size = 100;
    bogus.conv_cache = 3;
    bogus.conv_dim = 4096;
    bogus.max_position_embeddings = 4096;
    bogus.bos_token_id = 1;
    bogus.eos_token_id = 2;
    bogus.pad_token_id = 0;
    bogus.norm_eps = 1e-5f;
    bogus.rope_theta = 1'000'000.0f;
    bogus.rope_type = "default";
    bogus.layer_types = {lfm::LayerType::Convolution, lfm::LayerType::FullAttention,
                         lfm::LayerType::Convolution, lfm::LayerType::FullAttention};
    bogus.compute_derived();
    bool rejected = false;
    try {
        lfm::ModelVariantRegistry::instance().select(bogus);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    std::filesystem::remove(path);
    std::cout << "config_test: ok\n";
}
