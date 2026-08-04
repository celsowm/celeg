#include "celeg/backend/cuda/model.hpp"
#include "celeg/backend/cuda/concurrency.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "support/assertions.hpp"

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

} // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "celeg-granite-cuda-test";
    write_checkpoint(directory);
    celeg::CudaModelOptions options;
    options.cuda_graph = false;
    options.fast_attention = false;
    options.fused_projections = false;
    options.fused_residuals = false;
    options.allocate_local_kv_cache = true;
    celeg::GenerationConfig generation;
    generation.seed = 7;
    generation.top_k = 1;
    {
        celeg::CudaModel model(directory.string(), 32, options, generation);
        model.session().prefill({1, 3, 4});
        CELEG_TEST_CHECK(model.session().ready_for_decode());
        CELEG_TEST_CHECK(model.diagnostics().copy_logits().size() == 32);
        (void)model.session().decode();
        CELEG_TEST_CHECK(model.session().position() == 4);
    }
    {
        celeg::ConcurrentEngineOptions engine_options;
        engine_options.max_active_requests = 2;
        engine_options.max_batched_tokens = 16;
        // Force an initial mixed ragged wave with no finalized rows; the
        // following wave finalizes both requests and exercises the zero-row
        // projection path.
        engine_options.prefill_chunk_tokens = 1;
        engine_options.page_tokens = 4;
        engine_options.worker_thread = false;
        engine_options.packed_decode = true;
        engine_options.packed_min_batch = 1;
        engine_options.ragged_packed_prefill = true;
        engine_options.ragged_prefill_min_batch = 2;
        engine_options.prefix_cache = false;
        celeg::ConcurrentEngine engine(directory.string(), 32, options,
                                     engine_options);
        celeg::ConcurrentRequestOptions request;
        request.max_new_tokens = 2;
        request.eos_tokens = {31};
        request.generation.seed = 7;
        request.generation.top_k = 1;
        const auto first = engine.submit({1, 3, 4}, request);
        const auto second = engine.submit({2, 5}, request);
        (void)engine.step();
        const celeg::CudaAllocationScope steady_state_allocations;
        for (int step = 1; step < 32; ++step) {
            if (celeg::is_terminal(engine.status(first)) &&
                celeg::is_terminal(engine.status(second))) break;
            (void)engine.step();
        }
        const celeg::CudaAllocationSnapshot allocation_delta =
            steady_state_allocations.delta();
        CELEG_TEST_CHECK(allocation_delta.device_allocations == 0);
        CELEG_TEST_CHECK(allocation_delta.host_allocations == 0);
        const celeg::PollResult first_result = engine.poll(first);
        const celeg::PollResult second_result = engine.poll(second);
        CELEG_TEST_CHECK(first_result.status == celeg::RequestStatus::Finished);
        CELEG_TEST_CHECK(second_result.status == celeg::RequestStatus::Finished);
        CELEG_TEST_CHECK(first_result.tokens.size() == 2);
        CELEG_TEST_CHECK(second_result.tokens.size() == 2);
        CELEG_TEST_CHECK(engine.metrics().packed_decode_tokens > 0);
        CELEG_TEST_CHECK(engine.metrics().ragged_prefill_tokens > 0);
        CELEG_TEST_CHECK(engine.release(first));
        CELEG_TEST_CHECK(engine.release(second));
    }
    std::filesystem::remove_all(directory);
    return 0;
}
