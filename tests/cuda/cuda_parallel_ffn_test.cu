#include "backend/cuda/concurrency.hpp"
#include "backend/cuda/model.hpp"
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

void add_zero_tensor(std::vector<Tensor>& tensors, std::string name,
                     std::vector<int> shape) {
    Tensor tensor{std::move(name), std::move(shape), {}};
    tensor.values.resize(elements(tensor.shape), bf16(0.0f));
    tensors.push_back(std::move(tensor));
}

/// Parallel-branch mode: 0 omits the branch, 1 binds it with bitwise-zero
/// weights, 2 binds live weights. A non-null `hidden_act` overrides the
/// dense activation (e.g. `"gelu_tanh"`).
void write_checkpoint(const std::filesystem::path& directory, int parallel_mode,
                      const char* hidden_act = nullptr) {
    std::filesystem::create_directories(directory);
    std::ofstream config(directory / "config.json");
    config << R"({
      "model_type":"granite", "torch_dtype":"bfloat16",
      "hidden_size":32, "intermediate_size":64, "num_hidden_layers":1,
      "num_attention_heads":2, "num_key_value_heads":1, "vocab_size":32,
      "max_position_embeddings":64, "bos_token_id":1, "eos_token_id":2,
      "pad_token_id":0, "rms_norm_eps":1e-5, "rope_theta":10000.0,
      "embedding_multiplier":2.0, "attention_multiplier":0.3535533906,
      "residual_multiplier":0.5, "logits_scaling":2.0,
      "tie_word_embeddings":true)";
    if (parallel_mode > 0) {
        config << ",\n      \"parallel_ffn_intermediate_size\":16";
    }
    if (hidden_act != nullptr) {
        config << ",\n      \"hidden_activation\":\"" << hidden_act << "\"";
    }
    config << "\n    }";

    std::vector<Tensor> tensors;
    add_tensor(tensors, "model.embed_tokens.weight", {32, 32}, 0.01f);
    add_tensor(tensors, "model.norm.weight", {32}, 1.0f);
    add_tensor(tensors, "model.layers.0.input_layernorm.weight", {32}, 1.0f);
    add_tensor(tensors, "model.layers.0.post_attention_layernorm.weight", {32}, 1.0f);
    add_tensor(tensors, "model.layers.0.self_attn.q_proj.weight", {32, 32}, 0.02f);
    add_tensor(tensors, "model.layers.0.self_attn.k_proj.weight", {16, 32}, 0.02f);
    add_tensor(tensors, "model.layers.0.self_attn.v_proj.weight", {16, 32}, 0.02f);
    add_tensor(tensors, "model.layers.0.self_attn.o_proj.weight", {32, 32}, 0.02f);
    add_tensor(tensors, "model.layers.0.mlp.gate_proj.weight", {64, 32}, 0.02f);
    add_tensor(tensors, "model.layers.0.mlp.up_proj.weight", {64, 32}, 0.02f);
    add_tensor(tensors, "model.layers.0.mlp.down_proj.weight", {32, 64}, 0.02f);
    if (parallel_mode == 1) {
        add_zero_tensor(tensors, "model.layers.0.mlp.parallel_ffn.gate_proj.weight", {16, 32});
        add_zero_tensor(tensors, "model.layers.0.mlp.parallel_ffn.up_proj.weight", {16, 32});
        add_zero_tensor(tensors, "model.layers.0.mlp.parallel_ffn.down_proj.weight", {32, 16});
    } else if (parallel_mode == 2) {
        add_tensor(tensors, "model.layers.0.mlp.parallel_ffn.gate_proj.weight", {16, 32}, 0.5f);
        add_tensor(tensors, "model.layers.0.mlp.parallel_ffn.up_proj.weight", {16, 32}, 0.5f);
        add_tensor(tensors, "model.layers.0.mlp.parallel_ffn.down_proj.weight", {32, 16}, 0.5f);
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

float max_abs_diff(const std::vector<float>& left, const std::vector<float>& right) {
    CELEG_TEST_CHECK(left.size() == right.size());
    float maximum = 0.0f;
    for (std::size_t i = 0; i < left.size(); ++i) {
        maximum = std::max(maximum, std::abs(left[i] - right[i]));
    }
    return maximum;
}

struct RunResult {
    std::vector<float> prefill_logits;
    std::vector<int32_t> decoded;
};

RunResult run_model(const std::string& directory, bool fused_projections,
                    bool fused_residuals) {
    celeg::CudaModelOptions options;
    options.cuda_graph = false;
    options.fast_attention = false;
    options.fused_projections = fused_projections;
    options.fused_residuals = fused_residuals;
    options.allocate_local_kv_cache = true;
    celeg::GenerationConfig generation;
    generation.seed = 7;
    generation.top_k = 1;
    celeg::CudaModel model(directory, 32, options, generation);
    model.session().prefill({1, 3, 5, 7});
    RunResult result;
    result.prefill_logits = model.diagnostics().copy_logits();
    result.decoded.push_back(model.session().decode());
    result.decoded.push_back(model.session().decode());
    return result;
}

}

int main() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() / "celeg-parallel-ffn-cuda-test";
    try {
        write_checkpoint(base / "absent", 0);
        write_checkpoint(base / "zero", 1);
        write_checkpoint(base / "live", 2);
        write_checkpoint(base / "gelu", 2, "gelu_tanh");

        /// Every fused/unfused combination must agree on the live checkpoint
        /// (exercising the parallel branch in decode and prefill under all
        /// four linear/residual code paths), the zero-weight branch must be
        /// an exact no-op per combination, and the live branch must move the
        /// logits per combination.
        RunResult reference;
        bool has_reference = false;
        for (const auto [fused_projections, fused_residuals] :
             {std::pair<bool, bool>{false, false},
              std::pair<bool, bool>{false, true},
              std::pair<bool, bool>{true, false},
              std::pair<bool, bool>{true, true}}) {
            const RunResult live =
                run_model((base / "live").string(), fused_projections, fused_residuals);
            const RunResult absent = run_model(
                (base / "absent").string(), fused_projections, fused_residuals);
            const RunResult zero = run_model(
                (base / "zero").string(), fused_projections, fused_residuals);
            CELEG_TEST_CHECK(max_abs_diff(absent.prefill_logits, zero.prefill_logits) < 1e-5f);
            CELEG_TEST_CHECK(absent.decoded == zero.decoded);
            CELEG_TEST_CHECK(max_abs_diff(absent.prefill_logits, live.prefill_logits) > 1e-4f);
            if (!has_reference) {
                reference = live;
                has_reference = true;
            } else {
                CELEG_TEST_CHECK(
                    max_abs_diff(reference.prefill_logits, live.prefill_logits) < 5e-3f);
                CELEG_TEST_CHECK(reference.decoded == live.decoded);
            }
        }

        /// The packed concurrent engine serves the parallel-branch model
        /// end to end and agrees with direct greedy decoding. The granite
        /// multipliers force `fused_residuals` off inside `configure_model`,
        /// so this also locks the executor/lane fingerprint agreement.
        /// The GeluTanh variant additionally covers the packed executor's
        /// gated-gelu-tanh paths (fused and split layouts).
        for (const char* checkpoint : {"live", "gelu"}) {
            const RunResult direct =
                run_model((base / checkpoint).string(), true, false);
            celeg::CudaModelOptions options;
            options.cuda_graph = false;
            options.fast_attention = false;
            options.fused_projections = true;
            options.fused_residuals = true;
            options.allocate_local_kv_cache = true;
            celeg::ConcurrentEngineOptions engine_options;
            engine_options.max_active_requests = 2;
            engine_options.max_batched_tokens = 16;
            engine_options.prefill_chunk_tokens = 1;
            engine_options.page_tokens = 4;
            engine_options.worker_thread = false;
            engine_options.packed_decode = true;
            engine_options.ragged_packed_prefill = true;
            engine_options.prefix_cache = false;
            celeg::ConcurrentEngine engine(
                (base / checkpoint).string(), 32, options, engine_options);
            celeg::ConcurrentRequestOptions request;
            request.max_new_tokens = 2;
            request.eos_tokens = {31};
            request.generation.seed = 7;
            request.generation.top_k = 1;
            const auto id = engine.submit({1, 3, 5, 7}, request);
            for (int step = 0; step < 32; ++step) {
                if (celeg::is_terminal(engine.status(id))) break;
                (void)engine.step();
            }
            const celeg::PollResult result = engine.poll(id);
            CELEG_TEST_CHECK(result.status == celeg::RequestStatus::Finished);
            CELEG_TEST_CHECK(result.tokens.size() == 2);
            CELEG_TEST_CHECK(result.tokens == direct.decoded);
            CELEG_TEST_CHECK(engine.release(id));
        }
    } catch (...) {
        std::filesystem::remove_all(base);
        throw;
    }
    std::filesystem::remove_all(base);
    return 0;
}
