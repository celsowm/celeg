#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/prefix_cache.hpp"
#include "support/assertions.hpp"

#include <algorithm>
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

std::size_t element_count(const std::vector<int>& shape) {
    std::size_t count = 1;
    for (int dimension : shape) count *= static_cast<std::size_t>(dimension);
    return count;
}

void add_tensor(std::vector<Tensor>& tensors, std::string name,
                std::vector<int> shape, float value) {
    Tensor tensor{std::move(name), std::move(shape), {}};
    tensor.values.resize(element_count(tensor.shape));
    for (std::size_t index = 0; index < tensor.values.size(); ++index) {
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
          "mrope_interleaved":true, "mrope_section":[1,1,1]},
        "linear_num_key_heads":1, "linear_num_value_heads":1,
        "linear_key_head_dim":8, "linear_value_head_dim":8,
        "linear_conv_kernel_dim":4, "tie_word_embeddings":true
      }
    })";

    std::vector<Tensor> tensors;
    add_tensor(tensors, "model.language_model.embed_tokens.weight", {248048, 24}, 0.01f);
    add_tensor(tensors, "model.language_model.norm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.input_layernorm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.post_attention_layernorm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.q_proj.weight", {48, 24}, 0.02f);
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

void write_shared_moe_checkpoint(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    std::ofstream config(directory / "config.json");
    config << R"({
      "model_type":"qwen3_5_moe", "torch_dtype":"bfloat16",
      "text_config":{
        "model_type":"qwen3_5_moe_text", "hidden_size":24,
        "intermediate_size":16, "moe_intermediate_size":12,
        "shared_expert_intermediate_size":10, "num_hidden_layers":1,
        "num_experts":2, "num_experts_per_tok":1, "vocab_size":248048,
        "max_position_embeddings":64, "bos_token_id":1, "eos_token_id":2,
        "pad_token_id":0, "rms_norm_eps":1e-5, "head_dim":24,
        "num_attention_heads":1, "num_key_value_heads":1,
        "num_hidden_layers":1, "layer_types":["full_attention"],
        "rope_parameters":{"rope_theta":10000.0,
          "mrope_interleaved":true, "mrope_section":[1,1,1]},
        "linear_num_key_heads":1, "linear_num_value_heads":1,
        "linear_key_head_dim":8, "linear_value_head_dim":8,
        "linear_conv_kernel_dim":4, "tie_word_embeddings":true
      }
    })";

    std::vector<Tensor> tensors;
    add_tensor(tensors, "model.language_model.embed_tokens.weight", {248048, 24}, 0.01f);
    add_tensor(tensors, "model.language_model.norm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.input_layernorm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.post_attention_layernorm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.q_proj.weight", {48, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.q_norm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.k_proj.weight", {24, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.k_norm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.v_proj.weight", {24, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.self_attn.o_proj.weight", {24, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.gate.weight", {2, 24}, 0.03f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.experts.gate_up_proj", {2, 24, 24}, 0.04f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.experts.down_proj", {2, 24, 12}, 0.05f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", {10, 24}, 0.06f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", {10, 24}, 0.07f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.shared_expert.down_proj.weight", {24, 10}, 0.08f);
    add_tensor(tensors, "model.language_model.layers.0.mlp.shared_expert_gate.weight", {1, 24}, 0.09f);

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

void write_gated_delta_checkpoint(const std::filesystem::path& directory) {
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
        "layer_types":["linear_attention"],
        "rope_parameters":{"rope_theta":10000.0},
        "linear_num_key_heads":1, "linear_num_value_heads":1,
        "linear_key_head_dim":8, "linear_value_head_dim":8,
        "linear_conv_kernel_dim":4, "tie_word_embeddings":true
      }
    })";

    std::vector<Tensor> tensors;
    add_tensor(tensors, "model.language_model.embed_tokens.weight", {248048, 24}, 0.01f);
    add_tensor(tensors, "model.language_model.norm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.input_layernorm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.post_attention_layernorm.weight", {24}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", {24, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.linear_attn.in_proj_z.weight", {8, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.linear_attn.in_proj_a.weight", {1, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.linear_attn.in_proj_b.weight", {1, 24}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.linear_attn.conv1d.weight", {24, 1, 4}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.linear_attn.dt_bias", {1}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.linear_attn.A_log", {1}, 0.02f);
    add_tensor(tensors, "model.language_model.layers.0.linear_attn.norm.weight", {8}, 1.0f);
    add_tensor(tensors, "model.language_model.layers.0.linear_attn.out_proj.weight", {24, 8}, 0.02f);
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

int run_test() {
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

    {
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
    }

    std::filesystem::remove_all(directory);

    const std::filesystem::path shared_moe_directory =
        std::filesystem::temp_directory_path() / "celeg-qwen35-shared-moe-cpu-test";
    write_shared_moe_checkpoint(shared_moe_directory);
    {
        options.prefill_chunk_threshold = 64;
        celeg::CpuModel scalar(shared_moe_directory.string(), 32, options, generation);
        scalar.session().prefill({1, 3, 4, 5, 6});
        const auto scalar_logits = scalar.diagnostics().copy_logits();
        const auto scalar_snapshot = scalar.persistence().export_prefix_snapshot();

        options.prefill_chunk_threshold = 1;
        options.prefill_chunk_tokens = 2;
        celeg::CpuModel chunked(shared_moe_directory.string(), 32, options, generation);
        chunked.session().prefill({1, 3, 4, 5, 6});
        compare_logits(scalar_logits, chunked.diagnostics().copy_logits());
        const auto chunked_snapshot = chunked.persistence().export_prefix_snapshot();
        CELEG_TEST_CHECK(chunked_snapshot.position == scalar_snapshot.position);
        CELEG_TEST_CHECK(chunked_snapshot.attention_token_counts ==
                         scalar_snapshot.attention_token_counts);
        CELEG_TEST_CHECK(chunked.session().decode() == scalar.session().decode());
    }
    std::filesystem::remove_all(shared_moe_directory);

    const std::filesystem::path gated_delta_directory =
        std::filesystem::temp_directory_path() / "celeg-qwen35-gated-delta-cpu-test";
    write_gated_delta_checkpoint(gated_delta_directory);
    {
        options.prefill_chunk_threshold = 64;
        celeg::CpuModel scalar(gated_delta_directory.string(), 32, options, generation);
        scalar.session().prefill({1, 3, 4, 5, 6});
        const auto scalar_logits = scalar.diagnostics().copy_logits();
        const auto scalar_snapshot = scalar.persistence().export_prefix_snapshot();

        options.prefill_chunk_threshold = 1;
        options.prefill_chunk_tokens = 2;
        celeg::CpuModel chunked(gated_delta_directory.string(), 32, options, generation);
        chunked.session().prefill({1, 3, 4, 5, 6});
        compare_logits(scalar_logits, chunked.diagnostics().copy_logits());
        CELEG_TEST_CHECK(chunked.persistence().export_prefix_snapshot().position ==
                         scalar_snapshot.position);
        CELEG_TEST_CHECK(chunked.session().decode() == scalar.session().decode());
    }
    std::filesystem::remove_all(gated_delta_directory);
    return 0;
}

int main() {
    try {
        return run_test();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
