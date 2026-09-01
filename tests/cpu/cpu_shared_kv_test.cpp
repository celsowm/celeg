#include "celeg/backend/cpu/model.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/runtime/context.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Tensor {
    std::string name;
    std::vector<int> shape;
    std::vector<std::uint16_t> values;
};

std::uint16_t bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x8000u) >> 16);
}

std::size_t element_count(const std::vector<int>& shape) {
    std::size_t count = 1;
    for (const int dimension : shape) {
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

void add_pattern_tensor(std::vector<Tensor>& tensors, std::string name,
                        std::vector<int> shape, float base, float step) {
    Tensor tensor{std::move(name), std::move(shape), {}};
    tensor.values.resize(element_count(tensor.shape));
    for (std::size_t index = 0; index < tensor.values.size(); ++index) {
        tensor.values[index] = bf16(
            base + static_cast<float>((index * 7 + 3) % 17) * step);
    }
    tensors.push_back(std::move(tensor));
}

void add_constant_tensor(std::vector<Tensor>& tensors, std::string name,
                         std::vector<int> shape, float value) {
    Tensor tensor{std::move(name), std::move(shape), {}};
    tensor.values.assign(element_count(tensor.shape), bf16(value));
    tensors.push_back(std::move(tensor));
}

void write_checkpoint(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    std::ofstream config(directory / "config.json");
    config << R"({"model_type":"celeg_cpu_shared_kv_fixture"})";

    std::vector<Tensor> tensors;
    add_pattern_tensor(tensors, "embed", {8, 4}, 0.02f, 0.003f);
    add_pattern_tensor(tensors, "l0.q", {4, 4}, 0.01f, 0.002f);
    add_pattern_tensor(tensors, "l0.k", {4, 4}, 0.03f, 0.0015f);
    add_pattern_tensor(tensors, "l0.v", {4, 4}, 0.04f, 0.0013f);
    add_constant_tensor(tensors, "l0.out", {4, 4}, 0.0f);
    add_pattern_tensor(tensors, "l1.q", {4, 4}, 0.05f, 0.0017f);
    add_pattern_tensor(tensors, "l1.k", {4, 4}, 0.03f, 0.0015f);
    add_pattern_tensor(tensors, "l1.v", {4, 4}, 0.04f, 0.0013f);
    add_pattern_tensor(tensors, "l1.out", {4, 4}, 0.02f, 0.0021f);

    std::ostringstream header;
    header << '{';
    std::size_t offset = 0;
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        if (index != 0) header << ',';
        const Tensor& tensor = tensors[index];
        header << '"' << tensor.name << "\":{";
        header << "\"dtype\":\"BF16\",\"shape\":[";
        for (std::size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
            if (dimension != 0) header << ',';
            header << tensor.shape[dimension];
        }
        header << "],\"data_offsets\":[" << offset << ','
               << offset + tensor.values.size() * sizeof(std::uint16_t) << "]}";
        offset += tensor.values.size() * sizeof(std::uint16_t);
    }
    header << '}';

    const std::string header_text = header.str();
    std::ofstream weights(directory / "model.safetensors", std::ios::binary);
    const std::uint64_t header_size = static_cast<std::uint64_t>(header_text.size());
    weights.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    weights.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const Tensor& tensor : tensors) {
        weights.write(reinterpret_cast<const char*>(tensor.values.data()),
                      static_cast<std::streamsize>(
                          tensor.values.size() * sizeof(std::uint16_t)));
    }
}

celeg::TensorRequest request(celeg::TensorRole role, int layer,
                             std::string name, std::vector<int64_t> shape) {
    celeg::TensorRequest result;
    result.role = role;
    result.layer = layer;
    result.expected_shape = std::move(shape);
    result.source_name = std::move(name);
    return result;
}

class FixtureArchitecture final : public celeg::IArchitecture {
public:
    explicit FixtureArchitecture(bool shared) : shared_(shared) {}

    std::string_view id() const override {
        return shared_ ? "cpu_shared_kv_fixture" : "cpu_private_kv_fixture";
    }

    celeg::ProbeResult probe(const celeg::CheckpointMetadata&) const override {
        return {true, 1000000, "CPU shared-KV execution fixture"};
    }

    celeg::ResolvedModel resolve(const celeg::CheckpointView&) const override {
        celeg::ResolvedModel model;
        model.provenance.architecture_id = std::string(id());
        model.provenance.identity = std::string(id());
        model.provenance.source_format = "safetensors";
        model.graph.hidden = 4;
        model.graph.final_norm = celeg::NormSpec{
            1.0e-5f, celeg::NormWeightKind::None,
            celeg::NormGranularity::WholeVector};
        model.graph.tied_embeddings = true;
        model.graph.layers.resize(2);

        for (celeg::LayerSpec& layer : model.graph.layers) {
            celeg::AttentionSpec attention;
            attention.query_heads = 1;
            attention.key_value_heads = 1;
            attention.head_dim = 4;
            attention.query_scale = 0.5f;
            attention.position = celeg::NoPositionEncodingSpec{};
            attention.pattern = celeg::FullCausalPattern{};
            layer.mixer = attention;
            layer.feed_forward = std::monostate{};
        }
        if (shared_) {
            std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer).kv_sharing =
                celeg::SharedKvPublisher{0};
            std::get<celeg::AttentionSpec>(model.graph.layers[1].mixer).kv_sharing =
                celeg::SharedKvConsumer{0};
        }

        celeg::CheckpointDimensions dimensions;
        dimensions.vocab_size = 8;
        dimensions.max_position_embeddings = 32;
        dimensions.token_policy.bos_token_id = 0;
        dimensions.token_policy.eos_token_ids = {7};
        dimensions.token_policy.pad_token_id = 0;
        model.topology = celeg::compose_runtime_topology(
            std::move(dimensions), model.graph);

        model.weight_plan.requests = {
            request(celeg::TensorRole::TokenEmbedding, -1, "embed", {8, 4}),
            request(celeg::TensorRole::AttentionQuery, 0, "l0.q", {4, 4}),
            request(celeg::TensorRole::AttentionKey, 0, "l0.k", {4, 4}),
            request(celeg::TensorRole::AttentionValue, 0, "l0.v", {4, 4}),
            request(celeg::TensorRole::AttentionOutput, 0, "l0.out", {4, 4}),
            request(celeg::TensorRole::AttentionQuery, 1, "l1.q", {4, 4}),
            request(celeg::TensorRole::AttentionOutput, 1, "l1.out", {4, 4}),
        };
        if (!shared_) {
            model.weight_plan.requests.push_back(
                request(celeg::TensorRole::AttentionKey, 1, "l1.k", {4, 4}));
            model.weight_plan.requests.push_back(
                request(celeg::TensorRole::AttentionValue, 1, "l1.v", {4, 4}));
        }
        return model;
    }

private:
    bool shared_ = false;
};

std::shared_ptr<const celeg::RuntimeContext> fixture_runtime(bool shared) {
    celeg::RuntimeBuilder builder;
    builder.add_builtins();
    builder.add_architecture(std::make_unique<FixtureArchitecture>(shared));
    return builder.build_shared();
}

void compare_logits(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    CELEG_TEST_CHECK(expected.size() == actual.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        CELEG_TEST_CHECK(std::abs(expected[index] - actual[index]) < 1.0e-4f);
    }
}

}

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "celeg-cpu-shared-kv-test";
    write_checkpoint(directory);

    celeg::CpuModelOptions scalar_options;
    scalar_options.use_pack_cache = false;
    scalar_options.threads = 1;
    scalar_options.prefill_chunk_tokens = 2;
    scalar_options.prefill_chunk_threshold = 64;

    celeg::CpuModelOptions chunk_options = scalar_options;
    chunk_options.prefill_chunk_threshold = 1;

    celeg::GenerationConfig generation;
    generation.seed = 11;
    generation.top_k = 1;

    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};

    try {
        celeg::CpuModel shared_scalar(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime(true));
        celeg::CpuModel private_scalar(
            directory.string(), 32, scalar_options, generation,
            fixture_runtime(false));
        celeg::CpuModel shared_chunk(
            directory.string(), 32, chunk_options, generation,
            fixture_runtime(true));
        celeg::CpuModel private_chunk(
            directory.string(), 32, chunk_options, generation,
            fixture_runtime(false));

        CELEG_TEST_CHECK(shared_scalar.shared_kv_pools().size() == 1);
        CELEG_TEST_CHECK(private_scalar.shared_kv_pools().size() == 2);

        shared_scalar.session().prefill(prompt);
        private_scalar.session().prefill(prompt);
        shared_chunk.session().prefill(prompt);
        private_chunk.session().prefill(prompt);

        compare_logits(private_scalar.diagnostics().copy_logits(),
                       shared_scalar.diagnostics().copy_logits());
        compare_logits(private_chunk.diagnostics().copy_logits(),
                       shared_chunk.diagnostics().copy_logits());
        compare_logits(shared_scalar.diagnostics().copy_logits(),
                       shared_chunk.diagnostics().copy_logits());
        compare_logits(private_scalar.diagnostics().copy_logits(),
                       private_chunk.diagnostics().copy_logits());

        const int32_t shared_scalar_token = shared_scalar.session().decode();
        const int32_t private_scalar_token = private_scalar.session().decode();
        const int32_t shared_chunk_token = shared_chunk.session().decode();
        const int32_t private_chunk_token = private_chunk.session().decode();
        CELEG_TEST_CHECK(shared_scalar_token == private_scalar_token);
        CELEG_TEST_CHECK(shared_scalar_token == shared_chunk_token);
        CELEG_TEST_CHECK(shared_scalar_token == private_chunk_token);

        compare_logits(private_scalar.diagnostics().copy_logits(),
                       shared_scalar.diagnostics().copy_logits());
        compare_logits(private_chunk.diagnostics().copy_logits(),
                       shared_chunk.diagnostics().copy_logits());
        compare_logits(shared_scalar.diagnostics().copy_logits(),
                       shared_chunk.diagnostics().copy_logits());
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }

    std::filesystem::remove_all(directory);
    std::puts("cpu_shared_kv_test: ok");
    return 0;
}
