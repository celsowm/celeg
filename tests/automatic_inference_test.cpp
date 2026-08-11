#include "celeg/checkpoint/view.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "support/assertions.hpp"

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
    CELEG_TEST_CHECK(model.topology.hidden == 8);
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
    CELEG_TEST_CHECK(gguf_model.topology.hidden == 8);
    CELEG_TEST_CHECK(gguf_model.graph.layers.size() == 2);
    CELEG_TEST_CHECK(gguf_model.provenance.chat_template_id ==
                     "chat:thinking-function");
    CELEG_TEST_CHECK(celeg::explain_resolution(gguf_checkpoint).failures.empty());

    celeg::CheckpointView hybrid_checkpoint;
    hybrid_checkpoint.metadata = hybrid_gguf_metadata();
    hybrid_checkpoint.repository = hybrid_gguf_repository();
    const celeg::ResolvedModel hybrid_model =
        catalog.select(hybrid_checkpoint.metadata).resolve(hybrid_checkpoint);
    CELEG_TEST_CHECK(hybrid_model.topology.mixer_kinds[0] ==
                     celeg::MixerKind::ShortConvolution);
    CELEG_TEST_CHECK(hybrid_model.topology.mixer_kinds[1] == celeg::MixerKind::Attention);
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
    return 0;
}
