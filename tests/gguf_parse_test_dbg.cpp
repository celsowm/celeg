#include "lfm/checkpoint/formats/gguf.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Minimal little-endian GGUF v3 writer used to synthesize a self-contained
// fixture so the test does not depend on any downloaded checkpoint.
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
    std::cerr << "DEBUG: write_fixture start\n" << std::flush;
    GgufWriter w;
    w.put<uint32_t>(0x46554747u);  // magic "GGUF"
    w.put<uint32_t>(3);            // version
    w.put<uint64_t>(1);            // tensor count
    w.put<uint64_t>(4);            // kv count

    // KV: general.architecture = "lfm2" (string)
    w.put_str("general.architecture");
    w.put<uint32_t>(8);
    w.put_str("lfm2");
    // KV: lfm2.block_count = 14 (u32)
    w.put_str("lfm2.block_count");
    w.put<uint32_t>(4);
    w.put<uint32_t>(14);
    // KV: lfm2.rope.freq_base = 1e6 (f32)
    w.put_str("lfm2.rope.freq_base");
    w.put<uint32_t>(6);
    w.put<float>(1000000.0f);
    // KV: lfm2.attention.head_count_kv = [0,0,8] (i32 array)
    w.put_str("lfm2.attention.head_count_kv");
    w.put<uint32_t>(9);            // array
    w.put<uint32_t>(5);            // element kind i32
    w.put<uint64_t>(3);            // count
    w.put<int32_t>(0);
    w.put<int32_t>(0);
    w.put<int32_t>(8);

    // Tensor info: "token_embd.weight" F32, dims [4, 2] (GGUF order = cols,rows)
    w.put_str("token_embd.weight");
    w.put<uint32_t>(2);            // ndim
    w.put<uint64_t>(4);            // dim0 (cols)
    w.put<uint64_t>(2);            // dim1 (rows)
    w.put<int32_t>(0);
    std::cerr << "DEBUG: write_fixture after tensor info\n" << std::flush;
    w.put<uint64_t>(0);           // offset
    std::cerr << "DEBUG: write_fixture after offset\n" << std::flush;

    w.align(32);
    std::cerr << "DEBUG: write_fixture after align\n" << std::flush;
    const float payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    w.put_bytes(payload, sizeof(payload));
    std::cerr << "DEBUG: write_fixture after payload\n" << std::flush;

    const auto path = std::filesystem::temp_directory_path() / "lfm_gguf_fixture.gguf";
    std::ofstream out(path, std::ios::binary);
    std::cerr << "DEBUG: write_fixture writing file\n" << std::flush;
    out.write(reinterpret_cast<const char*>(w.buf.data()),
              static_cast<std::streamsize>(w.buf.size()));
    std::cerr << "DEBUG: write_fixture wrote " << w.buf.size() << " bytes\n" << std::flush;
    out.close();
    std::cerr << "DEBUG: write_fixture closing file\n" << std::flush;
    return path;
}

void test_fixture() {
    std::cerr << "DEBUG: test_fixture start\n" << std::flush;
    const auto path = write_fixture();
    {
        lfm::GgufFile g(path.string());

        assert(g.version() == 3);
        assert(g.str("general.architecture") == "lfm2");
        assert(g.u32("lfm2.block_count") == 14);
        assert(g.f32("lfm2.rope.freq_base") == 1000000.0f);

        const lfm::GgufValue& kv = g.value("lfm2.attention.head_count_kv");
        assert(kv.kind == lfm::GgufValueKind::Array);
        assert(kv.array_integers.size() == 3);
        assert(kv.array_integers[0] == 0 && kv.array_integers[2] == 8);

        assert(g.contains_tensor("token_embd.weight"));
        const lfm::GgufTensorView t = g.tensor("token_embd.weight");
        assert(t.type == lfm::GgmlType::F32);
        // HF shape is reversed GGUF dims: [rows=2, cols=4].
        assert((t.shape == std::vector<int64_t>{2, 4}));
        assert(t.element_count == 8);
        assert(t.bytes == 8 * sizeof(float));
        const float* data = reinterpret_cast<const float*>(t.data);
        assert(data[0] == 1.0f && data[7] == 8.0f);

        // Missing key / tensor should throw.
        bool threw = false;
        try { g.u32("does.not.exist"); } catch (const std::exception&) { threw = true; }
        assert(threw);
    }

    std::filesystem::remove(path);
    std::cout << "test_fixture PASS\n";
}

void test_real_file_optional() {
    const char* env = std::getenv("LFM_GGUF_TEST_FILE");
    if (env == nullptr) {
        std::cout << "test_real_file SKIP (set LFM_GGUF_TEST_FILE)\n";
        return;
    }
    lfm::GgufFile g(env);
    assert(g.str("general.architecture") == "lfm2");
    assert(g.contains_tensor("token_embd.weight"));
    assert(g.tensor_count() > 0);
    const auto t = g.tensor("blk.0.ffn_gate.weight");
    assert(t.type == lfm::GgmlType::Q4_K);
    std::cout << "test_real_file PASS (" << g.tensor_count() << " tensors)\n";
}

} // namespace

int main() {
    std::cerr << "DEBUG: main start\n" << std::flush;
    test_fixture();
    test_real_file_optional();
    std::cout << "ALL PASS\n";
    return 0;
}
