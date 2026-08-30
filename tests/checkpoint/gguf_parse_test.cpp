#include "celeg/checkpoint/formats/gguf.hpp"
#include "support/assertions.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <crtdbg.h>
#include <stdlib.h>
#endif

namespace {

struct GgufWriter {
    std::vector<std::byte> buf;

    template <typename T>
    void put(T value) {
        const auto* p = reinterpret_cast<const std::byte*>(&value);
        buf.insert(buf.end(), p, p + sizeof(T));
    }
    void put_str(const std::string& s) {
        put<uint64_t>(s.size());
        const auto* p = reinterpret_cast<const std::byte*>(s.data());
        buf.insert(buf.end(), p, p + s.size());
    }
    void put_bytes(const void* data, size_t n) {
        const auto* p = static_cast<const std::byte*>(data);
        buf.insert(buf.end(), p, p + n);
    }
    void align(size_t a) {
        while (buf.size() % a != 0) buf.push_back(std::byte{0});
    }
};

std::filesystem::path write_fixture() {
    GgufWriter w;
    w.put<uint32_t>(0x46554747u);
    w.put<uint32_t>(3);
    w.put<uint64_t>(1);
    w.put<uint64_t>(4);

    w.put_str("general.architecture");
    w.put<uint32_t>(8);
    w.put_str("lfm2");
    w.put_str("lfm2.block_count");
    w.put<uint32_t>(4);
    w.put<uint32_t>(14);
    w.put_str("lfm2.rope.freq_base");
    w.put<uint32_t>(6);
    w.put<float>(1000000.0f);
    w.put_str("lfm2.attention.head_count_kv");
    w.put<uint32_t>(9);
    w.put<uint32_t>(5);
    w.put<uint64_t>(3);
    w.put<int32_t>(0);
    w.put<int32_t>(0);
    w.put<int32_t>(8);

    w.put_str("token_embd.weight");
    w.put<uint32_t>(2);
    w.put<uint64_t>(4);
    w.put<uint64_t>(2);
    w.put<int32_t>(0);
    w.put<uint64_t>(0);

    w.align(32);
    const float payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    w.put_bytes(payload, sizeof(payload));

    const auto path = std::filesystem::temp_directory_path() / "celeg_gguf_fixture.gguf";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(w.buf.data()),
              static_cast<std::streamsize>(w.buf.size()));
    out.close();
    return path;
}

void test_fixture() {
    const auto path = write_fixture();
    {
        celeg::GgufFile g(path.string());

        CELEG_TEST_CHECK(g.version() == 3);
        CELEG_TEST_CHECK(g.str("general.architecture") == "lfm2");
        CELEG_TEST_CHECK(g.u32("lfm2.block_count") == 14);
        CELEG_TEST_CHECK(g.f32("lfm2.rope.freq_base") == 1000000.0f);

        const celeg::GgufValue& kv = g.value("lfm2.attention.head_count_kv");
        CELEG_TEST_CHECK(kv.kind == celeg::GgufValueKind::Array);
        CELEG_TEST_CHECK(kv.array_integers.size() == 3);
        CELEG_TEST_CHECK(kv.array_integers[0] == 0 && kv.array_integers[2] == 8);

        CELEG_TEST_CHECK(g.contains_tensor("token_embd.weight"));
        const celeg::GgufTensorView t = g.tensor("token_embd.weight");
        CELEG_TEST_CHECK(t.type == celeg::GgmlType::F32);
        CELEG_TEST_CHECK((t.shape == std::vector<int64_t>{2, 4}));
        CELEG_TEST_CHECK(t.element_count == 8);
        CELEG_TEST_CHECK(t.bytes == 8 * sizeof(float));
        const float* data = reinterpret_cast<const float*>(t.data);
        CELEG_TEST_CHECK(data[0] == 1.0f && data[7] == 8.0f);

        bool threw = false;
        try { g.u32("does.not.exist"); } catch (const std::exception&) { threw = true; }
        CELEG_TEST_CHECK(threw);
    }

    std::filesystem::remove(path);
    std::cout << "test_fixture PASS\n";
}

void test_q5k_trait() {
    const auto trait = celeg::ggml_type_trait(celeg::GgmlType::Q5_K);
    CELEG_TEST_CHECK(trait.block_size == 256);
    CELEG_TEST_CHECK(trait.type_size == 176);
    CELEG_TEST_CHECK(std::string(celeg::ggml_type_name(celeg::GgmlType::Q5_K)) == "Q5_K");
}

/// The type table drives parsing, geometry, naming and backend capability
/// all at once, so a row that is inconsistent with itself would surface as a
/// mysterious loader failure rather than a table error. Check the whole
/// table's invariants directly.
void test_type_registry() {
    struct Expected {
        celeg::GgmlType type;
        std::int32_t ordinal;
        const char* name;
        int block_size;
        int type_size;
    };
    // Ordinals and block geometry as ggml defines them; a divergence here
    // means celeg would read a real file at the wrong stride.
    const Expected expected[] = {
        {celeg::GgmlType::F32, 0, "F32", 1, 4},
        {celeg::GgmlType::F16, 1, "F16", 1, 2},
        {celeg::GgmlType::Q4_0, 2, "Q4_0", 32, 18},
        {celeg::GgmlType::Q4_1, 3, "Q4_1", 32, 20},
        {celeg::GgmlType::Q5_0, 6, "Q5_0", 32, 22},
        {celeg::GgmlType::Q8_0, 8, "Q8_0", 32, 34},
        {celeg::GgmlType::Q2_K, 10, "Q2_K", 256, 84},
        {celeg::GgmlType::Q3_K, 11, "Q3_K", 256, 110},
        {celeg::GgmlType::Q4_K, 12, "Q4_K", 256, 144},
        {celeg::GgmlType::Q5_K, 13, "Q5_K", 256, 176},
        {celeg::GgmlType::Q6_K, 14, "Q6_K", 256, 210},
        {celeg::GgmlType::IQ3_XXS, 18, "IQ3_XXS", 256, 98},
        {celeg::GgmlType::IQ4_NL, 20, "IQ4_NL", 32, 18},
        {celeg::GgmlType::IQ3_S, 21, "IQ3_S", 256, 110},
        {celeg::GgmlType::IQ2_S, 22, "IQ2_S", 256, 82},
        {celeg::GgmlType::IQ4_XS, 23, "IQ4_XS", 256, 136},
        {celeg::GgmlType::BF16, 30, "BF16", 1, 2},
    };
    for (const Expected& row : expected) {
        CELEG_TEST_CHECK(celeg::ggml_type_from_ordinal(row.ordinal) == row.type);
        CELEG_TEST_CHECK(std::string(celeg::ggml_type_name(row.type)) == row.name);
        const auto trait = celeg::ggml_type_trait(row.type);
        CELEG_TEST_CHECK(trait.block_size == row.block_size);
        CELEG_TEST_CHECK(trait.type_size == row.type_size);
        // Round-tripping through the block encoding is how the loaders carry
        // the type across the format/backend boundary.
        CELEG_TEST_CHECK(celeg::ggml_type_from_block_encoding(
            celeg::block_encoding_from_ggml_type(row.type)) == row.type);
    }

    // Unrecognised ordinals must degrade to Unknown with zero geometry, so
    // GgufFile::tensor() rejects the file instead of reading garbage.
    for (const std::int32_t ordinal : {7, 9, 15, 16, 17, 19, 29, 39, 1000, -5}) {
        const auto type = celeg::ggml_type_from_ordinal(ordinal);
        CELEG_TEST_CHECK(type == celeg::GgmlType::Unknown);
        CELEG_TEST_CHECK(celeg::ggml_type_trait(type).block_size == 0);
    }

    // Every quantized type must expose the neutral decoder, while dense GGML
    // types remain owned by their dtype loaders.
    for (const Expected& row : expected) {
        const bool quantized = celeg::ggml_type_trait(row.type).block_size > 1;
        CELEG_TEST_CHECK(celeg::ggml_row_decoder(row.type).has_value() == quantized);
    }
    CELEG_TEST_CHECK(!celeg::ggml_row_decoder(celeg::GgmlType::Unknown).has_value());
}

void test_real_file_optional() {
    const char* env = std::getenv("CELEG_GGUF_TEST_FILE");
    if (env == nullptr) {
        std::cout << "test_real_file SKIP (set CELEG_GGUF_TEST_FILE)\n";
        return;
    }
    celeg::GgufFile g(env);
    CELEG_TEST_CHECK(g.str("general.architecture") == "lfm2");
    CELEG_TEST_CHECK(g.contains_tensor("token_embd.weight"));
    CELEG_TEST_CHECK(g.tensor_count() > 0);
    const auto t = g.tensor("blk.0.ffn_gate.weight");
    CELEG_TEST_CHECK(t.type == celeg::GgmlType::Q4_K);
    std::cout << "test_real_file PASS (" << g.tensor_count() << " tensors)\n";
}

}

int main() {
#if defined(_WIN32)
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG);
#endif
    try {
        test_fixture();
        test_q5k_trait();
        test_type_registry();
        test_real_file_optional();
        std::cout << "ALL PASS\n";
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN EXCEPTION\n";
        return 1;
    }
    return 0;
}
