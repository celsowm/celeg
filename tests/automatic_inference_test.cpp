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

} // namespace

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
    CELEG_TEST_CHECK(model.topology.exec.hidden == 8);
    CELEG_TEST_CHECK(model.graph.embedding_transform.multiplier == 2.0f);
    CELEG_TEST_CHECK(std::abs(std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer).query_scale -
                              0.3535533906f) < 1.0e-6f);
    CELEG_TEST_CHECK(model.graph.layers[0].residual.multiplier == 0.5f);
    CELEG_TEST_CHECK(model.graph.logits_divisor == 2.0f);
    CELEG_TEST_CHECK(model.graph.layers.size() == 2);
    CELEG_TEST_CHECK(model.graph.layers.front().mixer_kind() == celeg::MixerKind::Attention);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::OrthogonalizeCurrentValueSpec>(
        std::get<celeg::AttentionSpec>(model.graph.layers.front().mixer).output_transform));
    CELEG_TEST_CHECK(celeg::explain_resolution(checkpoint).failures.empty());

    celeg::CheckpointView gguf_checkpoint;
    gguf_checkpoint.metadata = gguf_metadata();
    gguf_checkpoint.repository = gguf_repository();
    const auto& gguf_architecture = catalog.select(gguf_checkpoint.metadata);
    const celeg::ResolvedModel gguf_model = gguf_architecture.resolve(gguf_checkpoint);
    CELEG_TEST_CHECK(gguf_model.provenance.source_format == "gguf");
    CELEG_TEST_CHECK(gguf_model.topology.exec.hidden == 8);
    CELEG_TEST_CHECK(gguf_model.graph.layers.size() == 2);
    CELEG_TEST_CHECK(gguf_model.provenance.chat_template_id ==
                     "chat:thinking-function");
    CELEG_TEST_CHECK(celeg::explain_resolution(gguf_checkpoint).failures.empty());

    celeg::CheckpointView hybrid_checkpoint;
    hybrid_checkpoint.metadata = hybrid_gguf_metadata();
    hybrid_checkpoint.repository = hybrid_gguf_repository();
    const celeg::ResolvedModel hybrid_model =
        catalog.select(hybrid_checkpoint.metadata).resolve(hybrid_checkpoint);
    CELEG_TEST_CHECK(hybrid_model.topology.exec.mixer_kinds[0] ==
                     celeg::MixerKind::ShortConvolution);
    CELEG_TEST_CHECK(hybrid_model.topology.exec.mixer_kinds[1] == celeg::MixerKind::Attention);
    CELEG_TEST_CHECK(hybrid_model.capabilities.tied_embeddings);

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
    CELEG_TEST_CHECK(scoped_facts.key_value_heads.global == std::nullopt);
    CELEG_TEST_CHECK(scoped_facts.key_value_heads.value_for(0) == std::optional<int>{2});
    CELEG_TEST_CHECK(scoped_facts.key_value_heads.value_for(1) == std::optional<int>{1});

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

    auto ling_alias = metadata();
    ling_alias.values.erase("qk_norm");
    ling_alias.values["use_qk_norm"] = true;
    const auto ling_facts = celeg::normalize_model_metadata(ling_alias);
    CELEG_TEST_CHECK(ling_facts.query_key_norm == std::optional<bool>{true});

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
    CELEG_TEST_CHECK(ling_model.topology.exec.mixer_kinds[0] == celeg::MixerKind::GatedDeltaNet);
    CELEG_TEST_CHECK(ling_model.topology.exec.mixer_kinds[3] == celeg::MixerKind::Attention);
    CELEG_TEST_CHECK(ling_model.topology.exec.attention_layout(3).latent_state()->factorized);
    CELEG_TEST_CHECK(ling_model.topology.exec.feed_forward_kinds[0] == celeg::FeedForwardKind::Dense);
    CELEG_TEST_CHECK(ling_model.topology.exec.feed_forward_kinds[1] == celeg::FeedForwardKind::MixtureOfExperts);
    CELEG_TEST_CHECK(celeg::explain_resolution(ling_checkpoint).failures.empty());
    return 0;
}
