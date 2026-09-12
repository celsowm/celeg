#include "celeg/backend/cpu/model.hpp"
#include "support/assertions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
        tensor.values[index] = bf16(value + static_cast<float>(index % 7) * 0.001f);
    }
    tensors.push_back(std::move(tensor));
}

void add_zero_tensor(std::vector<Tensor>& tensors, std::string name,
                     std::vector<int> shape) {
    std::size_t count = 1;
    for (const int dimension : shape) count *= static_cast<std::size_t>(dimension);
    Tensor tensor{std::move(name), std::move(shape), {}};
    tensor.values.resize(count, bf16(0.0f));
    tensors.push_back(std::move(tensor));
}

/// Parallel-branch mode: 0 omits the branch, 1 binds it with bitwise-zero
/// weights (must reproduce the branch-less logits exactly), 2 binds live
/// weights (must move the logits, proving the branch executes). A non-null
/// `hidden_act` overrides the dense activation (e.g. `"gelu_tanh"`).
void write_checkpoint(const std::filesystem::path& directory, int parallel_mode,
                      const char* hidden_act = nullptr) {
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
      "tie_word_embeddings":true)";
    if (parallel_mode > 0) {
        config << ",\n      \"parallel_ffn_intermediate_size\":8";
    }
    if (hidden_act != nullptr) {
        config << ",\n      \"hidden_activation\":\"" << hidden_act << "\"";
    }
    config << "\n    }";

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
    if (parallel_mode == 1) {
        add_zero_tensor(tensors, "model.layers.0.mlp.parallel_ffn.gate_proj.weight", {8, 8});
        add_zero_tensor(tensors, "model.layers.0.mlp.parallel_ffn.up_proj.weight", {8, 8});
        add_zero_tensor(tensors, "model.layers.0.mlp.parallel_ffn.down_proj.weight", {8, 8});
    } else if (parallel_mode == 2) {
        add_tensor(tensors, "model.layers.0.mlp.parallel_ffn.gate_proj.weight", {8, 8}, 0.5f);
        add_tensor(tensors, "model.layers.0.mlp.parallel_ffn.up_proj.weight", {8, 8}, 0.5f);
        add_tensor(tensors, "model.layers.0.mlp.parallel_ffn.down_proj.weight", {8, 8}, 0.5f);
    }

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

float max_abs_diff(const std::vector<float>& left, const std::vector<float>& right) {
    CELEG_TEST_CHECK(left.size() == right.size());
    float maximum = 0.0f;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

struct RunResult {
    std::vector<float> prefill_logits;
    std::vector<float> decode_logits;
    int32_t decoded = -1;
};

RunResult run_prefill(const std::string& directory, bool chunked) {
    celeg::CpuModelOptions options;
    options.use_pack_cache = false;
    options.threads = 1;
    celeg::GenerationConfig generation;
    generation.seed = 7;
    generation.top_k = 1;
    const std::vector<int32_t> prompt = {1, 3, 4, 5, 6, 7};
    if (chunked) {
        options.prefill_chunk_threshold = 1;
        options.prefill_chunk_tokens = 3;
    } else {
        options.prefill_chunk_threshold = 64;
    }
    celeg::CpuModel model(directory, 32, options, generation);
    model.session().prefill(prompt);
    RunResult result;
    result.prefill_logits = model.diagnostics().copy_logits();
    result.decoded = model.session().decode();
    result.decode_logits = model.diagnostics().copy_logits();
    return result;
}

}

int main() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() / "celeg-parallel-ffn-cpu-test";
    try {
        write_checkpoint(base / "absent", 0);
        write_checkpoint(base / "zero", 1);
        write_checkpoint(base / "live", 2);
        write_checkpoint(base / "gelu", 2, "gelu_tanh");

        const RunResult absent = run_prefill((base / "absent").string(), false);
        const RunResult zero = run_prefill((base / "zero").string(), false);
        CELEG_TEST_CHECK(max_abs_diff(absent.prefill_logits, zero.prefill_logits) < 1e-6f);
        CELEG_TEST_CHECK(max_abs_diff(absent.decode_logits, zero.decode_logits) < 1e-6f);
        CELEG_TEST_CHECK(absent.decoded == zero.decoded);

        const RunResult live = run_prefill((base / "live").string(), false);
        CELEG_TEST_CHECK(max_abs_diff(absent.prefill_logits, live.prefill_logits) > 1e-4f);

        const RunResult chunked = run_prefill((base / "live").string(), true);
        CELEG_TEST_CHECK(max_abs_diff(live.prefill_logits, chunked.prefill_logits) < 1e-5f);
        CELEG_TEST_CHECK(live.decoded == chunked.decoded);

        /// Packed batch decode exercises the BatchScratch dense path
        /// (including the parallel branch) across two sessions at once.
        celeg::CpuModelOptions batch_options;
        batch_options.use_pack_cache = false;
        batch_options.threads = 1;
        batch_options.prefill_chunk_threshold = 64;
        celeg::GenerationConfig batch_generation;
        batch_generation.seed = 7;
        batch_generation.top_k = 1;
        celeg::CpuModel packed_owner(
            (base / "live").string(), 32, batch_options, batch_generation);
        std::unique_ptr<celeg::CpuModel> packed_a = packed_owner.clone_session();
        std::unique_ptr<celeg::CpuModel> packed_b = packed_owner.clone_session();
        const std::vector<int32_t> prompt = {1, 3, 4, 5, 6, 7};
        for (std::size_t index = 0; index < prompt.size(); ++index) {
            const bool final = index + 1 == prompt.size();
            const celeg::CpuPrefillItem items[] = {
                {packed_a.get(), prompt[index], final},
                {packed_b.get(), prompt[index], final},
            };
            CELEG_TEST_CHECK(celeg::CpuModel::prefill_batch(items).batch_size == 2);
        }
        CELEG_TEST_CHECK(
            max_abs_diff(live.prefill_logits, packed_a->diagnostics().copy_logits()) < 1e-5f);
        CELEG_TEST_CHECK(
            max_abs_diff(live.prefill_logits, packed_b->diagnostics().copy_logits()) < 1e-5f);
        celeg::CpuModel* packed_models[] = {packed_a.get(), packed_b.get()};
        const auto [packed_tokens, decode_metrics] =
            celeg::CpuModel::decode_batch(packed_models);
        CELEG_TEST_CHECK(decode_metrics.batch_size == 2);
        CELEG_TEST_CHECK(packed_tokens.size() == 2);
        CELEG_TEST_CHECK(packed_tokens[0] == live.decoded);
        CELEG_TEST_CHECK(packed_tokens[1] == live.decoded);

        /// GeluTanh activation with a live parallel branch must agree
        /// across the scalar, chunked, and packed-batch paths.
        const RunResult gelu_scalar = run_prefill((base / "gelu").string(), false);
        const RunResult gelu_chunked = run_prefill((base / "gelu").string(), true);
        CELEG_TEST_CHECK(
            max_abs_diff(gelu_scalar.prefill_logits, gelu_chunked.prefill_logits) < 1e-5f);
        CELEG_TEST_CHECK(gelu_scalar.decoded == gelu_chunked.decoded);
        celeg::CpuModel gelu_owner(
            (base / "gelu").string(), 32, batch_options, batch_generation);
        std::unique_ptr<celeg::CpuModel> gelu_a = gelu_owner.clone_session();
        std::unique_ptr<celeg::CpuModel> gelu_b = gelu_owner.clone_session();
        for (std::size_t index = 0; index < prompt.size(); ++index) {
            const bool final = index + 1 == prompt.size();
            const celeg::CpuPrefillItem gelu_items[] = {
                {gelu_a.get(), prompt[index], final},
                {gelu_b.get(), prompt[index], final},
            };
            CELEG_TEST_CHECK(celeg::CpuModel::prefill_batch(gelu_items).batch_size == 2);
        }
        CELEG_TEST_CHECK(
            max_abs_diff(gelu_scalar.prefill_logits, gelu_a->diagnostics().copy_logits()) <
            1e-5f);
        celeg::CpuModel* gelu_models[] = {gelu_a.get(), gelu_b.get()};
        const auto [gelu_tokens, gelu_metrics] = celeg::CpuModel::decode_batch(gelu_models);
        CELEG_TEST_CHECK(gelu_metrics.batch_size == 2);
        CELEG_TEST_CHECK(gelu_tokens[0] == gelu_scalar.decoded);
        CELEG_TEST_CHECK(gelu_tokens[1] == gelu_scalar.decoded);
    } catch (...) {
        std::filesystem::remove_all(base);
        throw;
    }
    std::filesystem::remove_all(base);
    return 0;
}
