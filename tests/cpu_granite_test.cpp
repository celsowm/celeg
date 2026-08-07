#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/prefix_cache.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <cmath>
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

std::size_t elements(const std::vector<int>& shape) {
    std::size_t count = 1;
    for (int dimension : shape) count *= static_cast<std::size_t>(dimension);
    return count;
}

void add_tensor(std::vector<Tensor>& tensors, std::string name,
                std::vector<int> shape, float value) {
    Tensor tensor{std::move(name), std::move(shape), {}};
    tensor.values.resize(elements(tensor.shape));
    for (std::size_t i = 0; i < tensor.values.size(); ++i) {
        tensor.values[i] = bf16(value + static_cast<float>(i % 7) * 0.001f);
    }
    tensors.push_back(std::move(tensor));
}

void write_checkpoint(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    std::ofstream config(directory / "config.json");
    config << R"({
      "model_type":"granite", "torch_dtype":"bfloat16",
      "hidden_size":8, "intermediate_size":16, "num_hidden_layers":1,
      "num_attention_heads":2, "num_key_value_heads":1, "vocab_size":32,
      "max_position_embeddings":64, "bos_token_id":1, "eos_token_id":2,
      "pad_token_id":0, "rms_norm_eps":1e-5, "rope_theta":10000.0,
      "embedding_multiplier":2.0, "attention_multiplier":0.3535533906,
      "residual_multiplier":0.5, "logits_scaling":2.0,
      "tie_word_embeddings":true
    })";

    std::vector<Tensor> tensors;
    add_tensor(tensors, "model.embed_tokens.weight", {32, 8}, 0.01f);
    add_tensor(tensors, "model.norm.weight", {8}, 1.0f);
    add_tensor(tensors, "model.layers.0.input_layernorm.weight", {8}, 1.0f);
    add_tensor(tensors, "model.layers.0.post_attention_layernorm.weight", {8}, 1.0f);
    add_tensor(tensors, "model.layers.0.self_attn.q_proj.weight", {8, 8}, 0.02f);
    add_tensor(tensors, "model.layers.0.self_attn.k_proj.weight", {4, 8}, 0.02f);
    add_tensor(tensors, "model.layers.0.self_attn.v_proj.weight", {4, 8}, 0.02f);
    add_tensor(tensors, "model.layers.0.self_attn.o_proj.weight", {8, 8}, 0.02f);
    add_tensor(tensors, "model.layers.0.mlp.gate_proj.weight", {16, 8}, 0.02f);
    add_tensor(tensors, "model.layers.0.mlp.up_proj.weight", {16, 8}, 0.02f);
    add_tensor(tensors, "model.layers.0.mlp.down_proj.weight", {8, 16}, 0.02f);

    std::ostringstream header;
    header << "{";
    std::size_t offset = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (i != 0) header << ',';
        const Tensor& tensor = tensors[i];
        header << '"' << tensor.name << "\":{";
        header << "\"dtype\":\"BF16\",\"shape\":[";
        for (std::size_t d = 0; d < tensor.shape.size(); ++d) {
            if (d != 0) header << ',';
            header << tensor.shape[d];
        }
        header << "],\"data_offsets\":[" << offset << ','
               << offset + tensor.values.size() * sizeof(std::uint16_t) << "]}";
        offset += tensor.values.size() * sizeof(std::uint16_t);
    }
    header << "}";

    std::ofstream weights(directory / "model.safetensors", std::ios::binary);
    const std::uint64_t header_size = static_cast<std::uint64_t>(header.str().size());
    weights.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    const std::string header_text = header.str();
    weights.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const Tensor& tensor : tensors) {
        weights.write(reinterpret_cast<const char*>(tensor.values.data()),
                      static_cast<std::streamsize>(tensor.values.size() * sizeof(std::uint16_t)));
    }
}

void write_moe_checkpoint(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    std::ofstream config(directory / "config.json");
    config << R"({
      "model_type":"lfm2_moe", "torch_dtype":"bfloat16",
      "hidden_size":8, "intermediate_size":16, "moe_intermediate_size":12,
      "num_hidden_layers":1, "num_attention_heads":2, "num_key_value_heads":1,
      "vocab_size":32, "conv_L_cache":3, "max_position_embeddings":64,
      "norm_eps":1e-5, "rope_theta":10000.0, "bos_token_id":1,
      "eos_token_id":2, "pad_token_id":0, "layer_types":["conv"],
      "num_dense_layers":0, "num_experts":3, "num_experts_per_tok":2,
      "norm_topk_prob":true, "use_expert_bias":true,
      "routed_scaling_factor":0.75
    })";

    std::vector<Tensor> tensors;
    add_tensor(tensors, "model.embed_tokens.weight", {32, 8}, 0.01f);
    add_tensor(tensors, "model.embedding_norm.weight", {8}, 1.0f);
    add_tensor(tensors, "model.layers.0.operator_norm.weight", {8}, 1.0f);
    add_tensor(tensors, "model.layers.0.ffn_norm.weight", {8}, 1.0f);
    add_tensor(tensors, "model.layers.0.conv.in_proj.weight", {24, 8}, 0.02f);
    add_tensor(tensors, "model.layers.0.conv.conv.weight", {8, 1, 3}, 0.02f);
    add_tensor(tensors, "model.layers.0.conv.out_proj.weight", {8, 8}, 0.02f);
    add_tensor(tensors, "model.layers.0.feed_forward.gate.weight", {3, 8}, 0.03f);
    add_tensor(tensors, "model.layers.0.feed_forward.expert_bias.weight", {3}, 0.04f);
    for (int expert = 0; expert < 3; ++expert) {
        const std::string prefix = "model.layers.0.feed_forward.experts." +
            std::to_string(expert) + ".";
        add_tensor(tensors, prefix + "w1.weight", {12, 8}, 0.05f + expert * 0.01f);
        add_tensor(tensors, prefix + "w3.weight", {12, 8}, 0.06f + expert * 0.01f);
        add_tensor(tensors, prefix + "w2.weight", {8, 12}, 0.07f + expert * 0.01f);
    }

    std::ostringstream header;
    header << "{";
    std::size_t offset = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (i != 0) header << ',';
        const Tensor& tensor = tensors[i];
        header << '"' << tensor.name << "\":{";
        header << "\"dtype\":\"BF16\",\"shape\":[";
        for (std::size_t d = 0; d < tensor.shape.size(); ++d) {
            if (d != 0) header << ',';
            header << tensor.shape[d];
        }
        header << "],\"data_offsets\":[" << offset << ','
               << offset + tensor.values.size() * sizeof(std::uint16_t) << "]}";
        offset += tensor.values.size() * sizeof(std::uint16_t);
    }
    header << "}";

    std::ofstream weights(directory / "model.safetensors", std::ios::binary);
    const std::uint64_t header_size = static_cast<std::uint64_t>(header.str().size());
    weights.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    const std::string header_text = header.str();
    weights.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const Tensor& tensor : tensors) {
        weights.write(reinterpret_cast<const char*>(tensor.values.data()),
                      static_cast<std::streamsize>(tensor.values.size() * sizeof(std::uint16_t)));
    }
}

} // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "celeg-granite-cpu-test";
    write_checkpoint(directory);
    celeg::CpuModelOptions options;
    options.use_pack_cache = false;
    options.threads = 1;
    celeg::GenerationConfig generation;
    generation.seed = 7;
    generation.top_k = 1;
    const std::vector<int32_t> prompt = {1, 3, 4, 5, 6, 7};
    {
    options.prefill_chunk_threshold = 64;
    celeg::CpuModel scalar(directory.string(), 32, options, generation);
    scalar.session().prefill(prompt);
    const auto scalar_logits = scalar.diagnostics().copy_logits();
    const auto scalar_snapshot = scalar.persistence().export_prefix_snapshot();

    options.prefill_chunk_threshold = 1;
    options.prefill_chunk_tokens = 3;
    celeg::CpuModel batched(directory.string(), 32, options, generation);
    batched.session().prefill(prompt);
    CELEG_TEST_CHECK(batched.session().ready_for_decode());
    const auto batched_logits = batched.diagnostics().copy_logits();
    const auto batched_snapshot = batched.persistence().export_prefix_snapshot();
    CELEG_TEST_CHECK(batched_logits.size() == scalar_logits.size());
    float max_logit_difference = 0.0f;
    for (size_t i = 0; i < scalar_logits.size(); ++i) {
        max_logit_difference = std::max(max_logit_difference,
            std::abs(batched_logits[i] - scalar_logits[i]));
    }
    CELEG_TEST_CHECK(max_logit_difference < 1e-5f);
    CELEG_TEST_CHECK(batched_snapshot.position == scalar_snapshot.position);
    CELEG_TEST_CHECK(batched_snapshot.attention_token_counts ==
                     scalar_snapshot.attention_token_counts);
    CELEG_TEST_CHECK(batched_snapshot.seen_tokens == scalar_snapshot.seen_tokens);

    options.prefill_chunk_tokens = 2;
    celeg::CpuModel boundary_two(directory.string(), 32, options, generation);
    boundary_two.session().prefill(prompt);
    const auto boundary_two_logits = boundary_two.diagnostics().copy_logits();
    for (size_t i = 0; i < scalar_logits.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(boundary_two_logits[i] - scalar_logits[i]) < 1e-5f);
    }
    CELEG_TEST_CHECK(boundary_two.persistence().export_prefix_snapshot().position ==
                     scalar_snapshot.position);
    CELEG_TEST_CHECK(boundary_two.persistence().export_prefix_snapshot().seen_tokens ==
                     scalar_snapshot.seen_tokens);

    const int32_t scalar_next = scalar.session().decode();
    const int32_t batched_next = batched.session().decode();
    CELEG_TEST_CHECK(batched_next == scalar_next);
    CELEG_TEST_CHECK(batched.session().position() ==
                     scalar.session().position());

    // Ragged prefill groups independent sessions through the layer-major
    // packed executor and must match independent scalar forwards.
    options.prefill_chunk_threshold = 64;
    celeg::CpuModel scalar_a(directory.string(), 32, options, generation);
    celeg::CpuModel scalar_b(directory.string(), 32, options, generation);
    scalar_a.session().prefill({8});
    scalar_b.session().prefill({9});
    const auto scalar_a_logits = scalar_a.diagnostics().copy_logits();
    const auto scalar_b_logits = scalar_b.diagnostics().copy_logits();
    celeg::CpuModel packed_a(directory.string(), 32, options, generation);
    auto packed_b = packed_a.clone_session();
    const std::vector<celeg::CpuPrefillItem> items = {
        {&packed_a, 8, true}, {packed_b.get(), 9, true}};
    celeg::CpuModel::prefill_batch(items);
    const auto packed_a_logits = packed_a.diagnostics().copy_logits();
    const auto packed_b_logits = packed_b->diagnostics().copy_logits();
    for (size_t i = 0; i < scalar_a_logits.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(packed_a_logits[i] - scalar_a_logits[i]) < 1e-5f);
        CELEG_TEST_CHECK(std::abs(packed_b_logits[i] - scalar_b_logits[i]) < 1e-5f);
    }
    CELEG_TEST_CHECK(packed_a.session().position() == scalar_a.session().position());
    CELEG_TEST_CHECK(packed_b->session().position() == scalar_b.session().position());
    }
    std::filesystem::remove_all(directory);

    const std::filesystem::path moe_directory =
        std::filesystem::temp_directory_path() / "celeg-moe-cpu-test";
    write_moe_checkpoint(moe_directory);
    {
    options.prefill_chunk_threshold = 64;
    celeg::CpuModel scalar(moe_directory.string(), 32, options, generation);
    scalar.session().prefill(prompt);
    const auto scalar_logits = scalar.diagnostics().copy_logits();
    const auto scalar_snapshot = scalar.persistence().export_prefix_snapshot();

    options.prefill_chunk_threshold = 1;
    options.prefill_chunk_tokens = 3;
    celeg::CpuModel batched(moe_directory.string(), 32, options, generation);
    batched.session().prefill(prompt);
    const auto batched_logits = batched.diagnostics().copy_logits();
    const auto batched_snapshot = batched.persistence().export_prefix_snapshot();
    for (size_t i = 0; i < scalar_logits.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(batched_logits[i] - scalar_logits[i]) < 1e-5f);
    }
    CELEG_TEST_CHECK(batched_snapshot.position == scalar_snapshot.position);
    CELEG_TEST_CHECK(batched_snapshot.attention_token_counts ==
                     scalar_snapshot.attention_token_counts);

    options.prefill_chunk_tokens = 2;
    celeg::CpuModel boundary_two(moe_directory.string(), 32, options, generation);
    boundary_two.session().prefill(prompt);
    const auto boundary_two_logits = boundary_two.diagnostics().copy_logits();
    for (size_t i = 0; i < scalar_logits.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(boundary_two_logits[i] - scalar_logits[i]) < 1e-5f);
    }
    CELEG_TEST_CHECK(boundary_two.persistence().export_prefix_snapshot().position ==
                     scalar_snapshot.position);
    CELEG_TEST_CHECK(batched.session().decode() == scalar.session().decode());

    options.prefill_chunk_threshold = 64;
    celeg::CpuModel packed_a(moe_directory.string(), 32, options, generation);
    auto packed_b = packed_a.clone_session();
    const std::vector<celeg::CpuPrefillItem> items = {
        {&packed_a, 8, true}, {packed_b.get(), 9, true}};
    celeg::CpuModel scalar_a(moe_directory.string(), 32, options, generation);
    celeg::CpuModel scalar_b(moe_directory.string(), 32, options, generation);
    scalar_a.session().prefill({8});
    scalar_b.session().prefill({9});
    celeg::CpuModel::prefill_batch(items);
    const auto packed_a_logits = packed_a.diagnostics().copy_logits();
    const auto packed_b_logits = packed_b->diagnostics().copy_logits();
    const auto scalar_a_logits = scalar_a.diagnostics().copy_logits();
    const auto scalar_b_logits = scalar_b.diagnostics().copy_logits();
    for (size_t i = 0; i < scalar_a_logits.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(packed_a_logits[i] - scalar_a_logits[i]) < 1e-5f);
        CELEG_TEST_CHECK(std::abs(packed_b_logits[i] - scalar_b_logits[i]) < 1e-5f);
    }
    }
    std::filesystem::remove_all(moe_directory);
    return 0;
}
