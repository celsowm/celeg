#include "celeg/checkpoint/view.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace {

class MemoryRepository final : public celeg::IWeightRepository {
public:
    void add(std::string name, std::vector<int64_t> shape) {
        shapes_.emplace(std::move(name), std::move(shape));
    }

    bool contains(std::string_view name) const override {
        return shapes_.contains(std::string(name));
    }
    celeg::HostTensorView tensor(std::string_view name) const override {
        const auto it = shapes_.find(std::string(name));
        if (it == shapes_.end()) throw std::out_of_range("missing synthetic tensor");
        return {celeg::TensorDType::BF16, it->second, nullptr, 0};
    }
    std::vector<std::string> names() const override {
        std::vector<std::string> result;
        for (const auto& [name, shape] : shapes_) {
            (void)shape;
            result.push_back(name);
        }
        return result;
    }

private:
    std::unordered_map<std::string, std::vector<int64_t>> shapes_;
};

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
    // Same structural/tensor grammar as gguf_metadata() (ordinary attention
    // tensors, via gguf_repository()), but declares a GGUF architecture that
    // is known to never apply RoPE despite carrying an active-looking
    // "rope.freq_base"/"rope.dimension_count". This mirrors the real
    // Nemotron-H bug: vestigial rope hparams inherited from a related
    // architecture family that the reference graph never consumes.
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
    // Qwen3.5's linear_attn geometry: key heads/dim independent from value
    // heads/dim, both distinct from num_attention_heads (full-attention
    // query heads).
    result.values["linear_num_key_heads"] = int64_t(2);
    result.values["linear_key_head_dim"] = int64_t(4);
    result.values["linear_num_value_heads"] = int64_t(3);
    result.values["linear_value_head_dim"] = int64_t(4);
    result.values["linear_conv_kernel_dim"] = int64_t(4);
    // Layer 0 is linear_attn, layer 1 is full_attention -- matches
    // qwen35_repository() below. Exercises the "linear_attention" token in
    // the generic per-layer attention-pattern parser (distinct from the
    // "gdn"/"gated_delta_net" synonyms it already recognized).
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
            // Output gate fused into q_proj (double width), as Qwen3.5's
            // full-attention layers store it.
            result->add(sa + "q_proj.weight", {64, 32});
            result->add(sa + "k_proj.weight", {32, 32});
            result->add(sa + "v_proj.weight", {32, 32});
            result->add(sa + "o_proj.weight", {32, 32});
        }
    }
    return result;
}

}

int main() {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = metadata();
    checkpoint.repository = repository();

    celeg::ArchitectureCatalog catalog;
    catalog.add(celeg::make_automatic_architecture());
    catalog.freeze();
    const auto& architecture = catalog.select(checkpoint.metadata);
    const celeg::ResolvedModel model = architecture.resolve(checkpoint);
    CELEG_TEST_CHECK(model.provenance.identity.find("automatic") != std::string::npos);
    CELEG_TEST_CHECK(model.graph.hidden == 8);
    CELEG_TEST_CHECK(model.graph.embedding_transform.multiplier == 2.0f);
    CELEG_TEST_CHECK(std::abs(std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer).query_scale -
                              0.3535533906f) < 1.0e-6f);
    CELEG_TEST_CHECK(model.graph.layers[0].residual.multiplier == 0.5f);
    CELEG_TEST_CHECK(model.graph.logits_divisor == 2.0f);
    CELEG_TEST_CHECK(model.graph.layers.size() == 2);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(
        model.graph.layers.front().mixer));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::OrthogonalizeCurrentValueSpec>(
        std::get<celeg::AttentionSpec>(model.graph.layers.front().mixer).output_transform));
    CELEG_TEST_CHECK(celeg::explain_resolution(checkpoint).failures.empty());

    celeg::CheckpointView gguf_checkpoint;
    gguf_checkpoint.metadata = gguf_metadata();
    gguf_checkpoint.repository = gguf_repository();
    const auto& gguf_architecture = catalog.select(gguf_checkpoint.metadata);
    const celeg::ResolvedModel gguf_model = gguf_architecture.resolve(gguf_checkpoint);
    CELEG_TEST_CHECK(gguf_model.provenance.source_format == "gguf");
    CELEG_TEST_CHECK(gguf_model.graph.hidden == 8);
    CELEG_TEST_CHECK(gguf_model.graph.layers.size() == 2);
    CELEG_TEST_CHECK(celeg::explain_resolution(gguf_checkpoint).failures.empty());
    // A GGUF architecture absent from the no-RoPE profile, with active
    // "rope.freq_base" metadata, must resolve to a real RopePositionSpec:
    // generic inference does not treat all GGUF checkpoints as position-free,
    // only the ones the format boundary declares as such.
    CELEG_TEST_CHECK(std::holds_alternative<celeg::RopePositionSpec>(
        std::get<celeg::AttentionSpec>(gguf_model.graph.layers[0].mixer).position));

    celeg::CheckpointView no_rope_checkpoint;
    no_rope_checkpoint.metadata = no_rope_gguf_metadata();
    no_rope_checkpoint.repository = gguf_repository();
    const celeg::ResolvedModel no_rope_model =
        catalog.select(no_rope_checkpoint.metadata).resolve(no_rope_checkpoint);
    // Same tensor grammar and same "active-looking" rope hparams as
    // gguf_model above, but a GGUF architecture the format boundary knows
    // never applies RoPE: the resolved attention layer must carry
    // NoPositionEncodingSpec regardless of the vestigial rope metadata.
    CELEG_TEST_CHECK(std::holds_alternative<celeg::NoPositionEncodingSpec>(
        std::get<celeg::AttentionSpec>(no_rope_model.graph.layers[0].mixer).position));
    CELEG_TEST_CHECK(celeg::explain_resolution(no_rope_checkpoint).failures.empty());

    auto tokenizer_vocab = gguf_metadata();
    tokenizer_vocab.values.erase("conventional.vocab_size");
    tokenizer_vocab.values["tokenizer.ggml.tokens"] =
        std::vector<std::string>(32, "token");
    celeg::CheckpointView tokenizer_vocab_checkpoint;
    tokenizer_vocab_checkpoint.metadata = std::move(tokenizer_vocab);
    tokenizer_vocab_checkpoint.repository = gguf_repository();
    const celeg::ResolvedModel tokenizer_vocab_model =
        catalog.select(tokenizer_vocab_checkpoint.metadata).resolve(tokenizer_vocab_checkpoint);
    CELEG_TEST_CHECK(tokenizer_vocab_model.topology.dims.vocab_size == 32);

    celeg::CheckpointView hybrid_checkpoint;
    hybrid_checkpoint.metadata = hybrid_gguf_metadata();
    hybrid_checkpoint.repository = hybrid_gguf_repository();
    const celeg::ResolvedModel hybrid_model =
        catalog.select(hybrid_checkpoint.metadata).resolve(hybrid_checkpoint);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::ShortConvolutionSpec>(
        hybrid_model.graph.layers[0].mixer));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(
        hybrid_model.graph.layers[1].mixer));
    CELEG_TEST_CHECK(hybrid_model.graph.tied_embeddings);

    auto conflicting = metadata();
    conflicting.values["n_embd"] = int64_t(9);
    bool rejected = false;
    try { (void)celeg::normalize_model_metadata(conflicting); }
    catch (const celeg::ResolutionError& error) {
        rejected = error.kind() == celeg::ResolutionFailureKind::ConflictingMetadata;
    }
    CELEG_TEST_CHECK(rejected);

    celeg::FactSolver solver;
    const auto proposal = solver.solve<int>({
        {{8}, {}, celeg::ProposalStrength::ExplicitMetadata, "a"},
        {{8}, {}, celeg::ProposalStrength::ShapeDerived, "b"}});
    CELEG_TEST_CHECK(proposal.value == 8);

    auto scoped = gguf_metadata();
    scoped.values["conventional.attention.head_count_kv"] =
        std::vector<int64_t>{2, 1};
    const auto scoped_facts = celeg::normalize_model_metadata(scoped);
    CELEG_TEST_CHECK(scoped_facts.attention.key_value_heads.global == std::nullopt);
    CELEG_TEST_CHECK(scoped_facts.attention.key_value_heads.value_for(0) == std::optional<int>{2});
    CELEG_TEST_CHECK(scoped_facts.attention.key_value_heads.value_for(1) == std::optional<int>{1});

    auto invalid_length = scoped;
    invalid_length.values["conventional.attention.head_count_kv"] =
        std::vector<int64_t>{2};
    bool invalid_length_rejected = false;
    try { (void)celeg::normalize_model_metadata(invalid_length); }
    catch (const celeg::ResolutionError& error) {
        invalid_length_rejected =
            error.kind() == celeg::ResolutionFailureKind::IncompleteLayerSchedule;
    }
    CELEG_TEST_CHECK(invalid_length_rejected);

    auto conflicting_scope = scoped;
    conflicting_scope.values["num_key_value_heads"] = int64_t(2);
    bool conflicting_scope_rejected = false;
    try { (void)celeg::normalize_model_metadata(conflicting_scope); }
    catch (const celeg::ResolutionError& error) {
        conflicting_scope_rejected =
            error.kind() == celeg::ResolutionFailureKind::ConflictingMetadata;
    }
    CELEG_TEST_CHECK(conflicting_scope_rejected);

    // NormalizedModelMetadata::position_encoding is the single canonical
    // representation of inferred positional semantics: a checkpoint carrying
    // "active-looking" rope hparams under a no-RoPE GGUF architecture must
    // resolve to NoPositionEncodingSpec (not a RoPE payload the runtime
    // happens to ignore), and a checkpoint under an ordinary GGUF
    // architecture with the same hparams must resolve to a real
    // InferredRopePosition carrying those values through unmodified.
    const auto no_rope_facts = celeg::normalize_model_metadata(no_rope_gguf_metadata());
    CELEG_TEST_CHECK(std::holds_alternative<celeg::NoPositionEncodingSpec>(
        no_rope_facts.attention.position_encoding));
    const auto rope_facts = celeg::normalize_model_metadata(gguf_metadata());
    CELEG_TEST_CHECK(std::holds_alternative<celeg::InferredRopePosition>(
        rope_facts.attention.position_encoding));
    CELEG_TEST_CHECK(std::get<celeg::InferredRopePosition>(rope_facts.attention.position_encoding).theta ==
        10000.0);

    auto ling_alias = metadata();
    ling_alias.values.erase("qk_norm");
    ling_alias.values["use_qk_norm"] = true;
    const auto ling_facts = celeg::normalize_model_metadata(ling_alias);
    CELEG_TEST_CHECK(ling_facts.attention.query_key_norm == std::optional<bool>{true});

    celeg::CheckpointView ling_checkpoint;
    ling_checkpoint.metadata = ling_metadata();
    ling_checkpoint.repository = ling_repository();
    celeg::ResolvedModel ling_model;
    try {
        ling_model = catalog.select(ling_checkpoint.metadata).resolve(ling_checkpoint);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ling failure: %s\n", error.what());
        return 1;
    }
    CELEG_TEST_CHECK(std::holds_alternative<celeg::GatedDeltaNetSpec>(
        ling_model.graph.layers[0].mixer));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(
        ling_model.graph.layers[3].mixer));
    CELEG_TEST_CHECK(std::get<celeg::AttentionSpec>(ling_model.graph.layers[3].mixer)
                         .latent_state()->factorized());
    CELEG_TEST_CHECK(std::holds_alternative<celeg::DenseFeedForwardSpec>(
        ling_model.graph.layers[0].feed_forward));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::MixtureOfExpertsSpec>(
        ling_model.graph.layers[1].feed_forward));
    CELEG_TEST_CHECK(celeg::explain_resolution(ling_checkpoint).failures.empty());

    // Qwen3.5: one linear_attn (gated-DeltaNet) layer followed by one
    // full-attention layer combining a q_proj-fused output gate, partial
    // rotary (0.25 of head_dim -> here 0.5, scaled for the tiny synthetic
    // head_dim), and interleaved M-RoPE sectioning -- the exact feature
    // combination Phase 2 needed to prove out for the generic/automatic
    // architecture path (no per-model descriptor).
    celeg::CheckpointView qwen35_checkpoint;
    qwen35_checkpoint.metadata = qwen35_metadata();
    qwen35_checkpoint.repository = qwen35_repository();
    celeg::ResolvedModel qwen35_model;
    try {
        qwen35_model = catalog.select(qwen35_checkpoint.metadata).resolve(qwen35_checkpoint);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "qwen3.5 failure: %s\n", error.what());
        return 1;
    }
    CELEG_TEST_CHECK(std::holds_alternative<celeg::GatedDeltaNetSpec>(
        qwen35_model.graph.layers[0].mixer));
    const celeg::GatedDeltaNetSpec& qwen35_delta =
        std::get<celeg::GatedDeltaNetSpec>(qwen35_model.graph.layers[0].mixer);
    CELEG_TEST_CHECK(qwen35_delta.key_heads == 2);
    CELEG_TEST_CHECK(qwen35_delta.key_head_dim == 4);
    CELEG_TEST_CHECK(qwen35_delta.value_heads == 3);
    CELEG_TEST_CHECK(qwen35_delta.value_head_dim == 4);
    CELEG_TEST_CHECK(qwen35_delta.conv_kernel == 4);

    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(
        qwen35_model.graph.layers[1].mixer));
    const celeg::AttentionSpec& qwen35_attention =
        std::get<celeg::AttentionSpec>(qwen35_model.graph.layers[1].mixer);
    CELEG_TEST_CHECK(qwen35_attention.output_gate.has_value());
    const auto* qwen35_mrope =
        std::get_if<celeg::MultiAxisRopeSpec>(&qwen35_attention.position);
    CELEG_TEST_CHECK(qwen35_mrope != nullptr);
    CELEG_TEST_CHECK(qwen35_mrope->interleaved);
    const std::array<int, 3> expected_sections{2, 1, 1};
    CELEG_TEST_CHECK(qwen35_mrope->sections == expected_sections);
    CELEG_TEST_CHECK(std::abs(qwen35_mrope->base.rotary_fraction - 0.5f) < 1.0e-6f);
    CELEG_TEST_CHECK(celeg::explain_resolution(qwen35_checkpoint).failures.empty());

    // Qwen3.5's `Qwen3_5RMSNorm` multiplies by `1 + weight`, not `weight`
    // directly (the checkpoint stores a zero-centered offset) -- detected
    // from the same `linear_attn.in_proj_qkv.weight` grammar that selects
    // the gated-delta layer above, so every structurally-bound norm in this
    // checkpoint (input/post-attention layernorm, q/k-norm, final norm)
    // must resolve to `OnePlusScale`.
    CELEG_TEST_CHECK(qwen35_model.graph.final_norm.weight_kind ==
                     celeg::NormWeightKind::OnePlusScale);
    CELEG_TEST_CHECK(qwen35_model.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(qwen35_model.graph.layers[0].mixer_norm.before->weight_kind ==
                     celeg::NormWeightKind::OnePlusScale);
    return 0;
}
