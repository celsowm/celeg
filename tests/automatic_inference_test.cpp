#include "celeg/checkpoint/view.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "support/assertions.hpp"

#include <memory>
#include <iostream>
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

} // namespace

int main() {
    std::cerr << "auto: start\n";
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = metadata();
    std::cerr << "auto: metadata built\n";
    checkpoint.repository = repository();
    std::cerr << "auto: repository built\n";

    celeg::ArchitectureCatalog catalog;
    catalog.add(celeg::make_automatic_architecture());
    catalog.freeze();
    std::cerr << "auto: catalog built\n";
    const auto& architecture = catalog.select(checkpoint.metadata);
    std::cerr << "auto: architecture selected\n";
    const celeg::ResolvedModel model = architecture.resolve(checkpoint);
    std::cerr << "auto: first resolved\n";
    CELEG_TEST_CHECK(model.provenance.identity.find("automatic") != std::string::npos);
    CELEG_TEST_CHECK(model.topology.hidden == 8);
    CELEG_TEST_CHECK(model.graph.layers.size() == 2);
    CELEG_TEST_CHECK(model.graph.layers.front().mixer_kind() == celeg::MixerKind::Attention);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::OrthogonalizeCurrentValueSpec>(
        std::get<celeg::AttentionSpec>(model.graph.layers.front().mixer).output_transform));
    CELEG_TEST_CHECK(celeg::explain_resolution(checkpoint).failures.empty());
    std::cerr << "auto: first explained\n";

    celeg::CheckpointView gguf_checkpoint;
    gguf_checkpoint.metadata = gguf_metadata();
    gguf_checkpoint.repository = gguf_repository();
    const auto& gguf_architecture = catalog.select(gguf_checkpoint.metadata);
    const celeg::ResolvedModel gguf_model = gguf_architecture.resolve(gguf_checkpoint);
    std::cerr << "auto: gguf resolved\n";
    CELEG_TEST_CHECK(gguf_model.provenance.source_format == "gguf");
    CELEG_TEST_CHECK(gguf_model.topology.hidden == 8);
    CELEG_TEST_CHECK(gguf_model.graph.layers.size() == 2);
    CELEG_TEST_CHECK(celeg::explain_resolution(gguf_checkpoint).failures.empty());
    std::cerr << "auto: gguf explained\n";

    auto conflicting = metadata();
    conflicting.values["n_embd"] = int64_t(9);
    bool rejected = false;
    try { (void)celeg::normalize_model_metadata(conflicting); }
    catch (const celeg::ResolutionError& error) {
        rejected = error.kind() == celeg::ResolutionFailureKind::ConflictingMetadata;
    }
    CELEG_TEST_CHECK(rejected);

    celeg::FactSolver solver;
    std::cerr << "auto: solver\n";
    const auto proposal = solver.solve<int>({
        {{8}, {}, celeg::ProposalStrength::ExplicitMetadata, "a"},
        {{8}, {}, celeg::ProposalStrength::ShapeDerived, "b"}});
    CELEG_TEST_CHECK(proposal.value == 8);
    return 0;
}
