#include "celeg/checkpoint/catalog.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/repositories/gguf.hpp"
#include "support/assertions.hpp"
#include "gguf_tensor_mapper.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct GgufWriter {
    std::vector<std::byte> bytes;

    template <typename T>
    void put(T value) {
        const auto* data = reinterpret_cast<const std::byte*>(&value);
        bytes.insert(bytes.end(), data, data + sizeof(T));
    }

    void put_string(const std::string& value) {
        put<uint64_t>(value.size());
        const auto* data = reinterpret_cast<const std::byte*>(value.data());
        bytes.insert(bytes.end(), data, data + value.size());
    }

    void put_string_array(const std::vector<std::string>& values) {
        put<uint32_t>(9); // array
        put<uint32_t>(8); // string
        put<uint64_t>(values.size());
        for (const std::string& value : values) put_string(value);
    }

    void put_i32_array(const std::vector<int32_t>& values) {
        put<uint32_t>(9); // array
        put<uint32_t>(5); // i32
        put<uint64_t>(values.size());
        for (const int32_t value : values) put(value);
    }

    void align(size_t alignment) {
        while (bytes.size() % alignment != 0) bytes.push_back(std::byte{0});
    }
};

struct TensorSpec {
    std::string name;
    std::vector<uint64_t> dimensions;
    std::vector<float> values;
};

std::filesystem::path write_fixture() {
    const std::vector<TensorSpec> tensors = {
        {"token_embd.weight", {2}, {1.0f, 2.0f}},
        {"blk.0.attn_q.weight", {4, 2}, {10.0f, 11.0f, 12.0f, 13.0f,
                                           14.0f, 15.0f, 16.0f, 17.0f}},
        {"blk.0.ffn_gate_exps.weight", {2, 2, 2}, {100.0f, 101.0f, 102.0f, 103.0f,
                                                     104.0f, 105.0f, 106.0f, 107.0f}},
    };

    GgufWriter writer;
    writer.put<uint32_t>(0x46554747u);
    writer.put<uint32_t>(3);
    writer.put<uint64_t>(tensors.size());
    writer.put<uint64_t>(8);

    writer.put_string("tokenizer.ggml.tokens");
    writer.put_string_array({"<pad>", "hello", "<eos>"});
    writer.put_string("tokenizer.ggml.merges");
    writer.put_string_array({"h e"});
    writer.put_string("tokenizer.ggml.token_type");
    writer.put_i32_array({3, 1, 3});
    writer.put_string("tokenizer.ggml.bos_token_id");
    writer.put<uint32_t>(4);
    writer.put<uint32_t>(1);
    writer.put_string("tokenizer.ggml.eos_token_id");
    writer.put<uint32_t>(4);
    writer.put<uint32_t>(2);
    writer.put_string("tokenizer.ggml.padding_token_id");
    writer.put<uint32_t>(4);
    writer.put<uint32_t>(0);
    writer.put_string("tokenizer.ggml.pre");
    writer.put<uint32_t>(8);
    writer.put_string("lfm2");
    writer.put_string("general.architecture");
    writer.put<uint32_t>(8);
    writer.put_string("lfm2");

    uint64_t offset = 0;
    for (const TensorSpec& tensor : tensors) {
        writer.put_string(tensor.name);
        writer.put<uint32_t>(tensor.dimensions.size());
        for (const uint64_t dimension : tensor.dimensions) writer.put(dimension);
        writer.put<int32_t>(0); // F32
        writer.put(offset);
        offset += tensor.values.size() * sizeof(float);
    }

    writer.align(32);
    for (const TensorSpec& tensor : tensors) {
        const auto* data = reinterpret_cast<const std::byte*>(tensor.values.data());
        writer.bytes.insert(writer.bytes.end(), data,
                            data + tensor.values.size() * sizeof(float));
    }

    const auto path = std::filesystem::temp_directory_path() /
        "celeg_gguf_repository_fixture.gguf";
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(writer.bytes.data()),
                 static_cast<std::streamsize>(writer.bytes.size()));
    return path;
}

} // namespace

int main() {
    const std::vector<std::pair<std::string_view, std::string_view>> mappings = {
        {"model.embed_tokens.weight", "token_embd.weight"},
        {"model.norm.weight", "output_norm.weight"},
        {"model.lm_head.weight", "output.weight"},
        {"model.layers.0.operator_norm.weight", "blk.0.attn_norm.weight"},
        {"model.layers.0.ffn_norm.weight", "blk.0.ffn_norm.weight"},
        {"model.layers.0.mixer.in_proj.weight", "blk.0.ssm_in.weight"},
        {"model.layers.0.mixer.conv1d.weight", "blk.0.ssm_conv1d.weight"},
        {"model.layers.0.mixer.dt_bias", "blk.0.ssm_dt.bias"},
        {"model.layers.0.mixer.A_log", "blk.0.ssm_a"},
        {"model.layers.0.mixer.out_proj.weight", "blk.0.ssm_out.weight"},
        {"model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"},
        {"model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight"},
        {"model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight"},
        {"model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"},
        {"model.layers.0.self_attn.q_layernorm.weight", "blk.0.attn_q_norm.weight"},
        {"model.layers.0.self_attn.k_layernorm.weight", "blk.0.attn_k_norm.weight"},
        {"model.layers.0.feed_forward.w1.weight", "blk.0.ffn_gate.weight"},
        {"model.layers.0.feed_forward.w3.weight", "blk.0.ffn_up.weight"},
        {"model.layers.0.feed_forward.w2.weight", "blk.0.ffn_down.weight"},
        {"model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"},
        {"model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"},
        {"model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"},
        {"model.layers.0.conv.in_proj.weight", "blk.0.shortconv.in_proj.weight"},
        {"model.layers.0.conv.conv.weight", "blk.0.shortconv.conv.weight"},
        {"model.layers.0.conv.out_proj.weight", "blk.0.shortconv.out_proj.weight"},
        {"model.layers.0.feed_forward.gate.weight", "blk.0.ffn_gate_inp.weight"},
        {"model.layers.0.feed_forward.expert_bias.weight", "blk.0.ffn_expert_bias.weight"},
    };
    for (const auto& [canonical, native] : mappings) {
        const celeg::GgufTensorReference reference =
            celeg::GgufTensorNameMapper::resolve(canonical);
        CELEG_TEST_CHECK(reference.native_name == native);
    }
    const celeg::GgufTensorReference expert_reference =
        celeg::GgufTensorNameMapper::resolve(
            "model.layers.0.feed_forward.experts.7.w3.weight");
    CELEG_TEST_CHECK(expert_reference.native_name == "blk.0.ffn_up_exps.weight");
    CELEG_TEST_CHECK(expert_reference.expert == 7);
    CELEG_TEST_CHECK(celeg::GgufTensorNameMapper::resolve(
                         "model.layers.0.unknown.weight").native_name.empty());

    const auto path = write_fixture();
    try {
        auto file = std::make_shared<celeg::GgufFile>(path.string());
        celeg::GgufRepository repository(file);

        CELEG_TEST_CHECK(repository.contains("model.embed_tokens.weight"));
        const celeg::HostTensorView embedding =
            repository.tensor("model.embed_tokens.weight");
        CELEG_TEST_CHECK(embedding.dtype == celeg::TensorDType::F32);
        CELEG_TEST_CHECK(embedding.shape == std::vector<int64_t>{2});

        const celeg::HostTensorView query = repository.tensor(
            "model.layers.0.self_attn.q_proj.weight");
        CELEG_TEST_CHECK(query.rows_rope_permuted);
        CELEG_TEST_CHECK(query.shape == std::vector<int64_t>({2, 4}));

        const celeg::HostTensorView expert = repository.tensor(
            "model.layers.0.feed_forward.experts.1.w1.weight");
        CELEG_TEST_CHECK(expert.shape == std::vector<int64_t>({2, 2}));
        CELEG_TEST_CHECK(reinterpret_cast<const float*>(expert.data)[0] == 104.0f);
        CELEG_TEST_CHECK(!repository.contains("model.layers.0.unknown.weight"));

        bool missing_threw = false;
        try { (void)repository.tensor("model.layers.0.unknown.weight"); }
        catch (const std::out_of_range&) { missing_threw = true; }
        CELEG_TEST_CHECK(missing_threw);

        auto catalog = celeg::create_builtin_checkpoint_format_catalog();
        const celeg::CheckpointView checkpoint = catalog->open(path);
        CELEG_TEST_CHECK(checkpoint.tokenizer != nullptr);
        CELEG_TEST_CHECK(checkpoint.tokenizer->tokens.size() == 3);
        CELEG_TEST_CHECK(checkpoint.tokenizer->bos_id == 1);
        CELEG_TEST_CHECK(checkpoint.tokenizer->eos_id == 2);
        CELEG_TEST_CHECK(checkpoint.tokenizer->pad_id == 0);
        CELEG_TEST_CHECK(checkpoint.tokenizer->pre_tokenizer ==
                         celeg::TokenizerData::PreTokenizerKind::NumericTriplets);
    } catch (...) {
        std::filesystem::remove(path);
        throw;
    }
    std::filesystem::remove(path);
    return 0;
}
