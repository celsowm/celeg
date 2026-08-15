#include "celeg/checkpoint/metadata.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/program.hpp"
#include "celeg/runtime/context.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class TestArchitecture final : public celeg::IArchitecture {
public:
    TestArchitecture(std::string name, int specificity)
        : name_(std::move(name)), specificity_(specificity) {}

    std::string_view id() const override { return name_; }
    celeg::ProbeResult probe(const celeg::CheckpointMetadata&) const override {
        return {true, specificity_, "test"};
    }
    celeg::ResolvedModel resolve(const celeg::CheckpointView&) const override { return {}; }

private:
    std::string name_;
    int specificity_;
};

celeg::CheckpointMetadata structural_metadata(std::string model_type) {
    celeg::CheckpointMetadata metadata;
    metadata.values["model_type"] = std::move(model_type);
    metadata.values["hidden_size"] = int64_t(576);
    metadata.values["intermediate_size"] = int64_t(1728);
    metadata.values["num_hidden_layers"] = int64_t(1);
    metadata.values["num_attention_heads"] = int64_t(9);
    metadata.values["num_key_value_heads"] = int64_t(3);
    metadata.values["head_dim"] = int64_t(64);
    metadata.values["vocab_size"] = int64_t(32770);
    metadata.values["max_position_embeddings"] = int64_t(8192);
    metadata.values["bos_token_id"] = int64_t(0);
    metadata.values["eos_token_id"] = int64_t(0);
    metadata.values["pad_token_id"] = int64_t(1);
    metadata.values["rms_norm_eps"] = 1.0e-6;
    metadata.values["rope_theta"] = 100000.0;
    metadata.values["tie_word_embeddings"] = true;
    return metadata;
}

class GptxRepository final : public celeg::IWeightRepository {
public:
    GptxRepository() {
        shapes_["transformer.wte.weight"] = {32770, 576};
        shapes_["transformer.ln_f.weight"] = {576};
        for (int layer = 0; layer < 1; ++layer) {
            const std::string prefix = "transformer.h." + std::to_string(layer);
            shapes_[prefix + ".ln_1.weight"] = {576};
            shapes_[prefix + ".attn.q_proj.weight"] = {576, 576};
            shapes_[prefix + ".attn.k_proj.weight"] = {192, 576};
            shapes_[prefix + ".attn.v_proj.weight"] = {192, 576};
            shapes_[prefix + ".attn.o_proj.weight"] = {576, 576};
            shapes_[prefix + ".ln_2.weight"] = {576};
            shapes_[prefix + ".mlp.w_gate.weight"] = {1728, 576};
            shapes_[prefix + ".mlp.w_up.weight"] = {1728, 576};
            shapes_[prefix + ".mlp.w_down.weight"] = {576, 1728};
        }
    }

    bool contains(std::string_view name) const override {
        return shapes_.contains(std::string(name));
    }

    celeg::HostTensorView tensor(std::string_view name) const override {
        return {celeg::TensorDType::BF16, shapes_.at(std::string(name)), nullptr, 0};
    }

    std::vector<std::string> names() const override {
        std::vector<std::string> result;
        result.reserve(shapes_.size());
        for (const auto& [name, shape] : shapes_) {
            (void)shape;
            result.push_back(name);
        }
        return result;
    }

private:
    std::unordered_map<std::string, std::vector<int64_t>> shapes_;
};

}

int main() {
    const auto runtime = celeg::create_builtin_runtime_context();
    const auto& catalog = runtime->architectures();
    CELEG_TEST_CHECK(catalog.find("automatic") != nullptr);

    for (const std::string model_type : {"lfm2", "qwen3_5", "granite", "gemma4"}) {
        const auto metadata = structural_metadata(model_type);
        CELEG_TEST_CHECK(catalog.select(metadata).id() == "automatic");
    }

    auto metadata = structural_metadata("gptx2");
    metadata.repository_hint = "AxiomicLabs/GPT-X2.5-135M";
    metadata.values["num_hidden_layers"] = int64_t(1);
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = metadata;
    checkpoint.repository = std::make_shared<GptxRepository>();
    const auto& architecture = catalog.select(metadata);
    const auto model = architecture.resolve(checkpoint);
    CELEG_TEST_CHECK(model.provenance.architecture_id == "automatic");
    CELEG_TEST_CHECK(model.graph.hidden == 576);
    CELEG_TEST_CHECK(model.graph.layers.size() == 1);
    CELEG_TEST_CHECK(model.capabilities.tied_embeddings);
    CELEG_TEST_CHECK(std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer).query_scale ==
                     0.125f);

    for (const auto& [model_type, architecture_id] : {
             std::pair<std::string, std::string>{
                 "celeg_topology_fixture_dense", "topology_fixture_dense"},
             std::pair<std::string, std::string>{
                 "celeg_topology_fixture_grouped_moe", "topology_fixture_grouped_moe"}}) {
        auto fixture_metadata = structural_metadata(model_type);
        celeg::CheckpointView fixture_checkpoint;
        fixture_checkpoint.metadata = std::move(fixture_metadata);
        const auto& fixture_architecture = catalog.select(fixture_checkpoint.metadata);
        CELEG_TEST_CHECK(fixture_architecture.id() == architecture_id);
        const auto fixture_model = fixture_architecture.resolve(fixture_checkpoint);
        const auto compiled = celeg::build_model_program(fixture_model);
        CELEG_TEST_CHECK(compiled.layers.size() == 4);
        if (architecture_id == "topology_fixture_dense") {
            CELEG_TEST_CHECK(std::get<celeg::DenseFeedForwardSpec>(
                                 fixture_model.graph.layers[2].feed_forward).intermediate_size == 8);
            CELEG_TEST_CHECK(fixture_model.graph.layers[2].post_attention_norm.has_value());
            CELEG_TEST_CHECK(std::get<celeg::CompiledDenseFeedForwardProgram>(
                                 compiled.layers[2].feed_forward).intermediate_size == 8);
        } else {
            const auto& moe = std::get<celeg::MixtureOfExpertsSpec>(
                fixture_model.graph.layers[2].feed_forward);
            const auto& grouped = std::get<celeg::MoeGroupedTopKSelectionSpec>(
                moe.selection);
            CELEG_TEST_CHECK(grouped.group_count == 2);
            CELEG_TEST_CHECK(grouped.experts_per_group == 2);
            CELEG_TEST_CHECK(grouped.groups_per_token == 1);
            const auto& compiled_moe = std::get<celeg::MoeLayerProgram>(
                compiled.layers[2].feed_forward);
            CELEG_TEST_CHECK(std::holds_alternative<celeg::MoeGroupedTopKSelectionSpec>(
                compiled_moe.router.selection));
        }
    }

    celeg::ArchitectureCatalog mutable_catalog;
    mutable_catalog.add(std::make_unique<TestArchitecture>("one", 10));
    bool duplicate_rejected = false;
    try {
        mutable_catalog.add(std::make_unique<TestArchitecture>("one", 10));
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    CELEG_TEST_CHECK(duplicate_rejected);
    mutable_catalog.freeze();
    bool mutation_rejected = false;
    try {
        mutable_catalog.add(std::make_unique<TestArchitecture>("two", 10));
    } catch (const std::logic_error&) {
        mutation_rejected = true;
    }
    CELEG_TEST_CHECK(mutation_rejected);

    celeg::ArchitectureCatalog ambiguous_catalog;
    ambiguous_catalog.add(std::make_unique<TestArchitecture>("one", 10));
    ambiguous_catalog.add(std::make_unique<TestArchitecture>("two", 10));
    bool ambiguity_rejected = false;
    try {
        ambiguous_catalog.select(metadata);
    } catch (const std::runtime_error&) {
        ambiguity_rejected = true;
    }
    CELEG_TEST_CHECK(ambiguity_rejected);

    std::cout << "architecture_resolution_test: ok\n";
}
