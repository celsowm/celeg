#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/prefix_cache.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
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

void add_tensor(std::vector<Tensor>& tensors, std::string name,
                std::vector<int> shape, float value) {
    std::size_t count = 1;
    for (const int dimension : shape) count *= static_cast<std::size_t>(dimension);
    Tensor tensor{std::move(name), std::move(shape), {}};
    tensor.values.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        tensor.values[index] = bf16(value + static_cast<float>(index % 11) * 0.0007f);
    }
    tensors.push_back(std::move(tensor));
}

void write_checkpoint(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    std::ofstream config(directory / "config.json");
    config << R"({
      "model_type":"qwen3_5", "torch_dtype":"bfloat16",
      "text_config":{
        "model_type":"qwen3_5_text", "hidden_size":24,
        "intermediate_size":32, "num_hidden_layers":1, "vocab_size":248048,
        "max_position_embeddings":64, "bos_token_id":1, "eos_token_id":2,
        "pad_token_id":0, "rms_norm_eps":1e-5, "head_dim":24,
        "num_attention_heads":1, "num_key_value_heads":1,
        "layer_types":["full_attention"],
        "rope_parameters":{"rope_theta":10000.0,
          "mrope_interleaved":true, "mrope_section":[4,4,4]},
        "tie_word_embeddings":true
      }
    })";

    std::vector<Tensor> tensors;
    add_tensor(tensors, "model.language_model.embed_tokens.weight", {248048, 24}, 0.01f);
    add_tensor(tensors, "model.language_model.norm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.input_layernorm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.post_attention_layernorm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.q_proj.weight", {24, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.q_norm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.k_proj.weight", {24, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.k_norm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.v_proj.weight", {24, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.o_proj.weight", {24, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.gate_proj.weight", {32, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.up_proj.weight", {32, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.down_proj.weight", {24, 32}, 0.02f);

    std::ostringstream header;
    header << "{";
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
    header << "}";
    const std::string header_text = header.str();
    std::ofstream weights(directory / "model.safetensors", std::ios::binary);
    const std::uint64_t header_size = static_cast<std::uint64_t>(header_text.size());
    weights.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    weights.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const Tensor& tensor : tensors) {
        weights.write(reinterpret_cast<const char*>(tensor.values.data()),
                      static_cast<std::streamsize>(tensor.values.size() * sizeof(std::uint16_t)));
    }
}

celeg::PromptEmbedding make_embeddings(std::size_t count) {
    celeg::PromptEmbedding embeddings;
    embeddings.width = 24;
    embeddings.positions = {2};
    embeddings.values.resize(24);
    for (std::size_t index = 0; index < embeddings.values.size(); ++index) {
        embeddings.values[index] = 0.03f + static_cast<float>(index) * 0.0003f;
    }
    embeddings.rope_positions.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int32_t position = static_cast<int32_t>(index);
        embeddings.rope_positions.push_back({
            position, index < 2 ? position : position + 7, position});
    }
    embeddings.next_rope_position = {
        static_cast<int32_t>(count), static_cast<int32_t>(count + 7),
        static_cast<int32_t>(count)};
    embeddings.has_rope_positions = true;
    return embeddings;
}

void compare_logits(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    CELEG_TEST_CHECK(expected.size() == actual.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        CELEG_TEST_CHECK(std::abs(expected[index] - actual[index]) < 1e-5f);
    }
}

} // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "celeg-qwen35-mrope-cpu-test";
    write_checkpoint(directory);

    celeg::CpuModelOptions options;
    options.use_pack_cache = false;
    options.threads = 1;
    options.prefill_chunk_tokens = 2;
    celeg::GenerationConfig generation;
    generation.seed = 7;
    generation.top_k = 1;
    const std::vector<int32_t> prompt = {1, 3, 4, 5, 6, 7};
    const celeg::PromptEmbedding embeddings = make_embeddings(prompt.size());

    try {
        options.prefill_chunk_threshold = 64;
        celeg::CpuModel scalar(directory.string(), 32, options, generation);
        scalar.session().prefill(prompt, embeddings);
        const auto scalar_logits = scalar.diagnostics().copy_logits();
        const auto scalar_snapshot = scalar.persistence().export_prefix_snapshot();

        options.prefill_chunk_threshold = 1;
        celeg::CpuModel chunked(directory.string(), 32, options, generation);
        chunked.session().prefill(prompt, embeddings);
        compare_logits(scalar_logits, chunked.diagnostics().copy_logits());
        const auto chunked_snapshot = chunked.persistence().export_prefix_snapshot();
        CELEG_TEST_CHECK(chunked_snapshot.position == scalar_snapshot.position);
        CELEG_TEST_CHECK(chunked_snapshot.attention_token_counts ==
                         scalar_snapshot.attention_token_counts);
        CELEG_TEST_CHECK(chunked.session().decode() == scalar.session().decode());
        CELEG_TEST_CHECK(chunked.session().position() == scalar.session().position());
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }

    std::filesystem::remove_all(directory);
    std::puts("cpu_mrope_test: ok");
    return 0;
}
