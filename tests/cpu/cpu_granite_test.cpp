#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/prefix_cache.hpp"
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

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
static LONG WINAPI diagnostic_vectored_handler(EXCEPTION_POINTERS* info) {
    fprintf(stderr, "DIAGNOSTIC CRASH exception 0x%08lX at 0x%p\n",
            info->ExceptionRecord->ExceptionCode,
            info->ExceptionRecord->ExceptionAddress);
    void* stack[64];
    const unsigned short frames =
        CaptureStackBackTrace(0, 64, stack, nullptr);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    for (unsigned short i = 0; i < frames; ++i) {
        char buf[sizeof(SYMBOL_INFO) + 256];
        SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 disp = 0;
        if (SymFromAddr(GetCurrentProcess(),
                        reinterpret_cast<DWORD64>(stack[i]), &disp, sym)) {
            fprintf(stderr, "#%u %s+0x%llx\n", i, sym->Name,
                    static_cast<unsigned long long>(disp));
        } else {
            fprintf(stderr, "#%u %p\n", i, stack[i]);
        }
    }
    fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif
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

void compare_logits(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    CELEG_TEST_CHECK(expected.size() == actual.size());
    float maximum = 0.0f;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        maximum = std::max(maximum, std::abs(expected[index] - actual[index]));
    }
    CELEG_TEST_CHECK(maximum < 1e-5f);
}

}

int main() {
#if defined(_WIN32)
    AddVectoredExceptionHandler(1, diagnostic_vectored_handler);
#endif
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "celeg-granite-cpu-test";
    write_checkpoint(directory);
    try {
        celeg::CpuModelOptions options;
        options.use_pack_cache = false;
        options.threads = 1;
        celeg::GenerationConfig generation;
        generation.seed = 7;
        generation.top_k = 1;
        const std::vector<int32_t> prompt = {1, 3, 4, 5, 6, 7};

        options.prefill_chunk_threshold = 64;
        celeg::CpuModel scalar(directory.string(), 32, options, generation);
        scalar.session().prefill(prompt);
        const auto scalar_logits = scalar.diagnostics().copy_logits();
        const auto scalar_snapshot = scalar.persistence().export_prefix_snapshot();

        options.prefill_chunk_threshold = 1;
        options.prefill_chunk_tokens = 3;
        celeg::CpuModel chunked(directory.string(), 32, options, generation);
        chunked.session().prefill(prompt);
        compare_logits(scalar_logits, chunked.diagnostics().copy_logits());
        const auto chunked_snapshot = chunked.persistence().export_prefix_snapshot();
        CELEG_TEST_CHECK(chunked_snapshot.position == scalar_snapshot.position);
        CELEG_TEST_CHECK(chunked_snapshot.attention_token_counts ==
                         scalar_snapshot.attention_token_counts);
        CELEG_TEST_CHECK(chunked_snapshot.seen_tokens == scalar_snapshot.seen_tokens);
        CELEG_TEST_CHECK(chunked.session().decode() == scalar.session().decode());
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
    return 0;
}
