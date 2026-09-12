#include "support.hpp"

#include "celeg/model/graph.hpp"
#include "celeg/model/weight_plan.hpp"
#include "support/assertions.hpp"

#include <stdexcept>

namespace celeg::automatic_inference_test {

celeg::CheckpointMetadata metadata() {
    celeg::CheckpointMetadata result;
    result.values["hidden_size"] = int64_t(8);
    result.values["intermediate_size"] = int64_t(16);
    result.values["num_hidden_layers"] = int64_t(2);
    result.values["num_attention_heads"] = int64_t(4);
    result.values["num_key_value_heads"] = int64_t(2);
    result.values["head_dim"] = int64_t(2);
    result.values["vocab_size"] = int64_t(32);
    result.values["max_position_embeddings"] = int64_t(64);
    result.values["rope_theta"] = 10000.0;
    result.values["xsa_projection"] = true;
    result.values["tie_word_embeddings"] = true;
    result.values["embedding_multiplier"] = 2.0;
    result.values["attention_multiplier"] = 0.3535533906;
    result.values["residual_multiplier"] = 0.5;
    result.values["logits_scaling"] = 2.0;
    return result;
}

std::shared_ptr<MemoryRepository> repository() {
    auto result = std::make_shared<MemoryRepository>();
    result->add("transformer.wte.weight", {32, 8});
    result->add("transformer.ln_f.weight", {8});
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "transformer.h." + std::to_string(layer);
        result->add(prefix + ".ln_1.weight", {8});
        result->add(prefix + ".attn.q_proj.weight", {8, 8});
        result->add(prefix + ".attn.k_proj.weight", {4, 8});
        result->add(prefix + ".attn.v_proj.weight", {4, 8});
        result->add(prefix + ".attn.o_proj.weight", {8, 8});
        result->add(prefix + ".ln_2.weight", {8});
        result->add(prefix + ".mlp.w_gate.weight", {16, 8});
        result->add(prefix + ".mlp.w_up.weight", {16, 8});
        result->add(prefix + ".mlp.w_down.weight", {8, 16});
    }
    return result;
}

celeg::CheckpointMetadata gguf_metadata() {
    celeg::CheckpointMetadata result;
    result.source_format = celeg::CheckpointSourceFormat::Gguf;
    result.values["general.architecture"] = std::string("conventional");
    result.values["conventional.embedding_length"] = int64_t(8);
    result.values["conventional.feed_forward_length"] = int64_t(16);
    result.values["conventional.block_count"] = int64_t(2);
    result.values["conventional.attention.head_count"] = int64_t(4);
    result.values["conventional.attention.head_count_kv"] = int64_t(2);
    result.values["conventional.attention.key_length"] = int64_t(2);
    result.values["conventional.vocab_size"] = int64_t(32);
    result.values["conventional.context_length"] = int64_t(64);
    result.values["conventional.attention.layer_norm_rms_epsilon"] = 1.0e-5;
    result.values["conventional.rope.freq_base"] = 10000.0;
    result.values["tokenizer.ggml.bos_token_id"] = int64_t(1);
    result.values["tokenizer.ggml.eos_token_id"] = int64_t(2);
    result.values["tokenizer.ggml.padding_token_id"] = int64_t(0);
    result.values["tokenizer.chat_template"] = std::string(
        "<|im_start|>{{ tools }}<|im_end|>{{ function }}");
    return result;
}

std::shared_ptr<MemoryRepository> gguf_repository() {
    auto result = std::make_shared<MemoryRepository>();
    result->add("token_embd.weight", {32, 8});
    result->add("output.weight", {32, 8});
    result->add("output_norm.weight", {8});
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer);
        result->add(prefix + ".attn_norm.weight", {8});
        result->add(prefix + ".attn_q.weight", {8, 8});
        result->add(prefix + ".attn_k.weight", {4, 8});
        result->add(prefix + ".attn_v.weight", {4, 8});
        result->add(prefix + ".attn_output.weight", {8, 8});
        result->add(prefix + ".ffn_norm.weight", {8});
        result->add(prefix + ".ffn_gate.weight", {16, 8});
        result->add(prefix + ".ffn_up.weight", {16, 8});
        result->add(prefix + ".ffn_down.weight", {8, 16});
    }
    return result;
}

celeg::CheckpointMetadata no_rope_gguf_metadata() {
    /// Same structural/tensor grammar as gguf_metadata() (ordinary attention
    /// tensors, via gguf_repository()), but declares a GGUF architecture that
    /// is known to never apply RoPE despite carrying an active-looking
    /// "rope.freq_base"/"rope.dimension_count". This mirrors the real
    /// Nemotron-H bug: vestigial rope hparams inherited from a related
    /// architecture family that the reference graph never consumes.
    celeg::CheckpointMetadata result = gguf_metadata();
    result.values["general.architecture"] = std::string("mamba2");
    for (const std::string_view suffix : {"embedding_length", "feed_forward_length",
                                          "block_count", "attention.head_count",
                                          "attention.head_count_kv", "attention.key_length",
                                          "vocab_size", "context_length",
                                          "attention.layer_norm_rms_epsilon", "rope.freq_base"}) {
        const std::string old_key = "conventional." + std::string(suffix);
        const std::string new_key = "mamba2." + std::string(suffix);
        result.values[new_key] = result.values.at(old_key);
        result.values.erase(old_key);
    }
    result.values["mamba2.rope.dimension_count"] = int64_t(2);
    return result;
}

celeg::CheckpointMetadata hybrid_gguf_metadata() {
    celeg::CheckpointMetadata result = gguf_metadata();
    result.values["general.architecture"] = std::string("hybrid");
    result.values.erase("conventional.attention.head_count_kv");
    result.values["hybrid.attention.head_count_kv"] = std::vector<int64_t>{0, 2};
    result.values["hybrid.shortconv.l_cache"] = int64_t(3);
    for (const std::string_view suffix : {"embedding_length", "feed_forward_length",
                                          "block_count", "attention.head_count",
                                          "attention.key_length", "vocab_size",
                                          "context_length", "attention.layer_norm_rms_epsilon",
                                          "rope.freq_base"}) {
        const std::string old_key = "conventional." + std::string(suffix);
        const std::string new_key = "hybrid." + std::string(suffix);
        result.values[new_key] = result.values.at(old_key);
        result.values.erase(old_key);
    }
    return result;
}

std::shared_ptr<MemoryRepository> hybrid_gguf_repository() {
    auto result = std::make_shared<MemoryRepository>();
    result->add("token_embd.weight", {32, 8});
    result->add("token_embd_norm.weight", {8});
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer);
        result->add(prefix + ".attn_norm.weight", {8});
        result->add(prefix + ".ffn_norm.weight", {8});
        result->add(prefix + ".ffn_gate.weight", {16, 8});
        result->add(prefix + ".ffn_up.weight", {16, 8});
        result->add(prefix + ".ffn_down.weight", {8, 16});
    }
    result->add("blk.0.shortconv.in_proj.weight", {24, 8});
    result->add("blk.0.shortconv.conv.weight", {8, 1, 3});
    result->add("blk.0.shortconv.out_proj.weight", {8, 8});
    result->add("blk.1.attn_q.weight", {8, 8});
    result->add("blk.1.attn_k.weight", {4, 8});
    result->add("blk.1.attn_v.weight", {4, 8});
    result->add("blk.1.attn_output.weight", {8, 8});
    result->add("blk.1.attn_q_norm.weight", {2});
    result->add("blk.1.attn_k_norm.weight", {2});
    return result;
}

celeg::CheckpointMetadata ling_metadata() {
    celeg::CheckpointMetadata result;
    result.values["model_type"] = std::string("bailing_hybrid");
    result.values["hidden_size"] = int64_t(1536);
    result.values["intermediate_size"] = int64_t(4608);
    result.values["num_hidden_layers"] = int64_t(24);
    result.values["num_attention_heads"] = int64_t(16);
    result.values["num_key_value_heads"] = int64_t(16);
    result.values["head_dim"] = int64_t(128);
    result.values["vocab_size"] = int64_t(157184);
    result.values["max_position_embeddings"] = int64_t(131072);
    result.values["rms_norm_eps"] = 1.0e-6;
    result.values["rope_theta"] = 6000000.0;
    result.values["partial_rotary_factor"] = 0.5;
    result.values["eos_token_id"] = int64_t(156895);
    result.values["pad_token_id"] = int64_t(156892);
    result.values["bos_token_id"] = int64_t(156892);
    result.values["tie_word_embeddings"] = false;
    result.values["first_k_dense_replace"] = int64_t(1);
    result.values["short_conv_kernel_size"] = int64_t(4);
    result.values["q_lora_rank"] = int64_t(256);
    result.values["kv_lora_rank"] = int64_t(512);
    result.values["qk_head_dim"] = int64_t(192);
    result.values["qk_nope_head_dim"] = int64_t(128);
    result.values["qk_rope_head_dim"] = int64_t(64);
    result.values["v_head_dim"] = int64_t(128);
    result.values["kda_safe_gate"] = true;
    result.values["kda_lower_bound"] = -5.0;
    result.values["num_experts"] = int64_t(128);
    result.values["num_experts_per_tok"] = int64_t(8);
    result.values["moe_intermediate_size"] = int64_t(512);
    result.values["moe_shared_expert_intermediate_size"] = int64_t(512);
    result.values["topk_group"] = int64_t(4);
    result.values["n_group"] = int64_t(8);
    result.values["routing_group_score_top_k"] = int64_t(2);
    result.values["norm_topk_prob"] = true;
    result.values["moe_router_enable_expert_bias"] = true;
    result.values["routed_scaling_factor"] = 2.5;
    result.values["scoring_func"] = std::string("sigmoid");
    result.values["tokenizer.chat_template"] = std::string(
        "<role>SYSTEM</role>{{ messages }}<|role_end|><arg_key>{{ key }}</arg_key><arg_value>{{ value }}</arg_value>");
    return result;
}

std::shared_ptr<MemoryRepository> ling_repository() {
    auto result = std::make_shared<MemoryRepository>();
    result->add("model.word_embeddings.weight", {157184, 1536});
    result->add("model.norm.weight", {1536});
    result->add("lm_head.weight", {157184, 1536});
    for (int layer = 0; layer < 24; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        result->add(prefix + ".input_layernorm.weight", {1536});
        result->add(prefix + ".post_attention_layernorm.weight", {1536});
        if (layer % 4 == 3) {
            result->add(prefix + ".attention.q_a_proj.weight", {256, 1536});
            result->add(prefix + ".attention.q_a_layernorm.weight", {256});
            result->add(prefix + ".attention.q_b_proj.weight", {3072, 256});
            result->add(prefix + ".attention.kv_a_proj_with_mqa.weight", {576, 1536});
            result->add(prefix + ".attention.kv_a_layernorm.weight", {512});
            result->add(prefix + ".attention.kv_b_proj.weight", {4096, 512});
            result->add(prefix + ".attention.g_proj.weight", {2048, 1536});
            result->add(prefix + ".attention.o_proj.weight", {1536, 2048});
        } else {
            result->add(prefix + ".attention.q_proj.weight", {2048, 1536});
            result->add(prefix + ".attention.k_proj.weight", {2048, 1536});
            result->add(prefix + ".attention.v_proj.weight", {2048, 1536});
            result->add(prefix + ".attention.f_proj.weight", {2048, 1536});
            result->add(prefix + ".attention.b_proj.weight", {16, 1536});
            result->add(prefix + ".attention.g_proj.weight", {2048, 1536});
            result->add(prefix + ".attention.q_conv1d.weight", {2048, 1, 4});
            result->add(prefix + ".attention.k_conv1d.weight", {2048, 1, 4});
            result->add(prefix + ".attention.v_conv1d.weight", {2048, 1, 4});
            result->add(prefix + ".attention.dt_bias", {2048});
            result->add(prefix + ".attention.A_log", {16});
            result->add(prefix + ".attention.o_norm.weight", {128});
            result->add(prefix + ".attention.o_proj.weight", {1536, 2048});
        }
        if (layer == 0) {
            result->add(prefix + ".mlp.gate_proj.weight", {4608, 1536});
            result->add(prefix + ".mlp.up_proj.weight", {4608, 1536});
            result->add(prefix + ".mlp.down_proj.weight", {1536, 4608});
        } else {
            result->add(prefix + ".mlp.gate.weight", {128, 1536});
            result->add(prefix + ".mlp.gate.expert_bias", {128});
            result->add(prefix + ".mlp.shared_experts.gate_proj.weight", {512, 1536});
            result->add(prefix + ".mlp.shared_experts.up_proj.weight", {512, 1536});
            result->add(prefix + ".mlp.shared_experts.down_proj.weight", {1536, 512});
            for (int expert = 0; expert < 128; ++expert) {
                const std::string expert_prefix = prefix + ".mlp.experts." +
                    std::to_string(expert);
                result->add(expert_prefix + ".gate_proj.weight", {512, 1536});
                result->add(expert_prefix + ".up_proj.weight", {512, 1536});
                result->add(expert_prefix + ".down_proj.weight", {1536, 512});
            }
        }
    }
    return result;
}

celeg::CheckpointMetadata qwen35_metadata() {
    celeg::CheckpointMetadata result;
    result.values["model_type"] = std::string("qwen3_5");
    result.values["hidden_size"] = int64_t(32);
    result.values["intermediate_size"] = int64_t(24);
    result.values["num_hidden_layers"] = int64_t(2);
    result.values["num_attention_heads"] = int64_t(2);
    result.values["num_key_value_heads"] = int64_t(2);
    result.values["head_dim"] = int64_t(16);
    result.values["vocab_size"] = int64_t(40);
    result.values["max_position_embeddings"] = int64_t(64);
    result.values["rms_norm_eps"] = 1.0e-6;
    result.values["rope_theta"] = 10000.0;
    result.values["partial_rotary_factor"] = 0.5;
    result.values["mrope_section"] = std::vector<int64_t>{2, 1, 1};
    result.values["mrope_interleaved"] = true;
    result.values["tie_word_embeddings"] = true;
    /// Qwen3.5's linear_attn geometry: key heads/dim independent from value
    /// heads/dim, both distinct from num_attention_heads (full-attention
    /// query heads).
    result.values["linear_num_key_heads"] = int64_t(2);
    result.values["linear_key_head_dim"] = int64_t(4);
    result.values["linear_num_value_heads"] = int64_t(3);
    result.values["linear_value_head_dim"] = int64_t(4);
    result.values["linear_conv_kernel_dim"] = int64_t(4);
    /// Layer 0 is linear_attn, layer 1 is full_attention -- matches
    /// qwen35_repository() below. Exercises the "linear_attention" token in
    /// the generic per-layer attention-pattern parser (distinct from the
    /// "gdn"/"gated_delta_net" synonyms it already recognized).
    result.values["layer_types"] =
        std::vector<std::string>{"linear_attention", "full_attention"};
    return result;
}

std::shared_ptr<MemoryRepository> qwen35_repository() {
    auto result = std::make_shared<MemoryRepository>();
    result->add("model.language_model.embed_tokens.weight", {40, 32});
    result->add("model.language_model.norm.weight", {32});
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer);
        result->add(prefix + ".input_layernorm.weight", {32});
        result->add(prefix + ".post_attention_layernorm.weight", {32});
        result->add(prefix + ".mlp.gate_proj.weight", {24, 32});
        result->add(prefix + ".mlp.up_proj.weight", {24, 32});
        result->add(prefix + ".mlp.down_proj.weight", {32, 24});
        if (layer == 0) {
            const std::string la = prefix + ".linear_attn.";
            result->add(la + "in_proj_qkv.weight", {28, 32});
            result->add(la + "in_proj_z.weight", {12, 32});
            result->add(la + "in_proj_a.weight", {3, 32});
            result->add(la + "in_proj_b.weight", {3, 32});
            result->add(la + "conv1d.weight", {28, 1, 4});
            result->add(la + "dt_bias", {3});
            result->add(la + "A_log", {3});
            result->add(la + "norm.weight", {4});
            result->add(la + "out_proj.weight", {32, 12});
        } else {
            const std::string sa = prefix + ".self_attn.";
            /// Output gate fused into q_proj (double width), as Qwen3.5's
            /// full-attention layers store it.
            result->add(sa + "q_proj.weight", {64, 32});
            result->add(sa + "k_proj.weight", {32, 32});
            result->add(sa + "v_proj.weight", {32, 32});
            result->add(sa + "o_proj.weight", {32, 32});
        }
    }
    return result;
}

/// Agnes-shaped hybrid metadata: one `delta_attn` gated-delta layer plus
/// one `global_attn` full-attention layer carrying a fused output gate,
/// with a parallel FFN branch on every dense layer. Mirrors the shipped
/// reference config keys (`parallel_ffn_intermediate_size`,
/// `global_attention_interval`, `attn_output_gate`, nested
/// `rope_parameters.rotary_fraction`).
celeg::CheckpointMetadata agnes_metadata() {
    celeg::CheckpointMetadata result;
    result.values["model_type"] = std::string("agnes");
    result.values["hidden_size"] = int64_t(32);
    result.values["intermediate_size"] = int64_t(24);
    result.values["parallel_ffn_intermediate_size"] = int64_t(8);
    result.values["num_hidden_layers"] = int64_t(2);
    result.values["num_attention_heads"] = int64_t(2);
    result.values["num_key_value_heads"] = int64_t(2);
    result.values["head_dim"] = int64_t(16);
    result.values["vocab_size"] = int64_t(40);
    result.values["max_position_embeddings"] = int64_t(64);
    result.values["rms_norm_eps"] = 1.0e-6;
    result.values["rope_theta"] = 10000.0;
    result.values["rope_parameters.rotary_fraction"] = 0.5;
    result.values["tie_word_embeddings"] = true;
    result.values["linear_num_key_heads"] = int64_t(2);
    result.values["linear_key_head_dim"] = int64_t(4);
    result.values["linear_num_value_heads"] = int64_t(3);
    result.values["linear_value_head_dim"] = int64_t(4);
    result.values["linear_conv_kernel_dim"] = int64_t(4);
    result.values["layer_types"] =
        std::vector<std::string>{"agnes_delta_attention", "agnes_global_attention"};
    result.values["global_attention_interval"] = int64_t(2);
    result.values["attn_output_gate"] = true;
    return result;
}

std::shared_ptr<MemoryRepository> agnes_repository(const std::string& layer_root,
                                                   const std::string& model_root) {
    auto result = std::make_shared<MemoryRepository>();
    result->add(model_root + "embed_tokens.weight", {40, 32});
    result->add(model_root + "norm.weight", {32});
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = layer_root + std::to_string(layer);
        result->add(prefix + ".input_layernorm.weight", {32});
        result->add(prefix + ".post_attention_layernorm.weight", {32});
        result->add(prefix + ".mlp.gate_proj.weight", {24, 32});
        result->add(prefix + ".mlp.up_proj.weight", {24, 32});
        result->add(prefix + ".mlp.down_proj.weight", {32, 24});
        result->add(prefix + ".mlp.parallel_ffn.gate_proj.weight", {8, 32});
        result->add(prefix + ".mlp.parallel_ffn.up_proj.weight", {8, 32});
        result->add(prefix + ".mlp.parallel_ffn.down_proj.weight", {32, 8});
        if (layer == 0) {
            const std::string da = prefix + ".delta_attn.";
            result->add(da + "in_proj_qkv.weight", {28, 32});
            result->add(da + "in_proj_z.weight", {12, 32});
            result->add(da + "in_proj_a.weight", {3, 32});
            result->add(da + "in_proj_b.weight", {3, 32});
            result->add(da + "conv1d.weight", {28, 1, 4});
            result->add(da + "dt_bias", {3});
            result->add(da + "A_log", {3});
            result->add(da + "norm.weight", {4});
            result->add(da + "out_proj.weight", {32, 12});
        } else {
            const std::string ga = prefix + ".global_attn.";
            result->add(ga + "q_proj.weight", {64, 32});
            result->add(ga + "k_proj.weight", {32, 32});
            result->add(ga + "v_proj.weight", {32, 32});
            result->add(ga + "o_proj.weight", {32, 32});
        }
    }
    return result;
}

void check_agnes_model(const celeg::ResolvedModel& model) {
    CELEG_TEST_CHECK(std::holds_alternative<celeg::GatedDeltaNetSpec>(
        model.graph.layers[0].mixer));
    const celeg::GatedDeltaNetSpec& delta =
        std::get<celeg::GatedDeltaNetSpec>(model.graph.layers[0].mixer);
    CELEG_TEST_CHECK(delta.key_heads == 2);
    CELEG_TEST_CHECK(delta.key_head_dim == 4);
    CELEG_TEST_CHECK(delta.value_heads == 3);
    CELEG_TEST_CHECK(delta.value_head_dim == 4);
    CELEG_TEST_CHECK(delta.conv_kernel == 4);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(
        model.graph.layers[1].mixer));
    CELEG_TEST_CHECK(std::get<celeg::AttentionSpec>(model.graph.layers[1].mixer)
                         .output_gate.has_value());
    for (int layer = 0; layer < 2; ++layer) {
        CELEG_TEST_CHECK(std::holds_alternative<celeg::DenseFeedForwardSpec>(
            model.graph.layers[static_cast<size_t>(layer)].feed_forward));
        const celeg::DenseFeedForwardSpec& dense = std::get<celeg::DenseFeedForwardSpec>(
            model.graph.layers[static_cast<size_t>(layer)].feed_forward);
        CELEG_TEST_CHECK(dense.intermediate_size == 24);
        CELEG_TEST_CHECK(dense.parallel_intermediate_size == 8);
    }
    const auto find_request = [&](celeg::TensorRole role, int layer)
        -> const celeg::TensorRequest& {
        for (const auto& request : model.weight_plan.requests) {
            if (request.role == role && request.layer == layer) return request;
        }
        throw std::runtime_error("agnes weight request missing");
    };
    CELEG_TEST_CHECK((find_request(celeg::TensorRole::FfnParallelGate, 0).expected_shape ==
                      std::vector<int64_t>{8, 32}));
    CELEG_TEST_CHECK((find_request(celeg::TensorRole::FfnParallelUp, 1).expected_shape ==
                      std::vector<int64_t>{8, 32}));
    CELEG_TEST_CHECK((find_request(celeg::TensorRole::FfnParallelDown, 1).expected_shape ==
                      std::vector<int64_t>{32, 8}));
}

}
