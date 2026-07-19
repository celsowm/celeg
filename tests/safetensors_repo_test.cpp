#include "lfm/safetensors.hpp"
#include "lfm/safetensors_repo.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void write_shard(const std::filesystem::path& path,
                 const std::string& header,
                 const void* data, size_t data_size) {
    const uint64_t header_size = header.size();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (data_size) {
        out.write(static_cast<const char*>(data),
                  static_cast<std::streamsize>(data_size));
    }
}

} // namespace

int main() {
 try {
    const auto dir = std::filesystem::temp_directory_path() / "lfm_repo_test";
    std::filesystem::create_directories(dir);
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        std::filesystem::remove(entry.path());
    }

    // Two shards: shard_0 holds "a" and "b"; shard_1 holds "c".
    const uint16_t a_vals[2] = {0x3f80, 0x4000}; // 1.0, 2.0
    const uint16_t b_vals[1] = {0x4040};          // 3.0
    const uint16_t c_vals[3] = {0x4080, 0x40a0, 0x40c0}; // 4,5,6
    {
        const std::string header =
            R"({"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}})";
        const uint64_t hs = header.size();
        std::ofstream out(dir / "model-00001-of-00002.safetensors", std::ios::binary);
        out.write(reinterpret_cast<const char*>(&hs), sizeof(hs));
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(reinterpret_cast<const char*>(a_vals), sizeof(a_vals));
        out.write(reinterpret_cast<const char*>(b_vals), sizeof(b_vals));
    }
    {
        const std::string header =
            R"({"c":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}})";
        const uint64_t hs = header.size();
        std::ofstream out(dir / "model-00002-of-00002.safetensors", std::ios::binary);
        out.write(reinterpret_cast<const char*>(&hs), sizeof(hs));
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(reinterpret_cast<const char*>(c_vals), sizeof(c_vals));
    }

    // Index mapping tensors to shards.
    const std::string index =
        R"({"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00001-of-00002.safetensors","c":"model-00002-of-00002.safetensors"}})";
    std::ofstream index_out(dir / "model.safetensors.index.json");
    index_out << index;
    index_out.close();

    {
        lfm::SafeTensorRepository repo(dir);
        assert(repo.sharded());
        assert(repo.contains("a"));
        assert(repo.contains("b"));
        assert(repo.contains("c"));
        assert(!repo.contains("missing"));

        std::vector<std::string> names = repo.names();
        assert(names.size() == 3);
        assert(names[0] == "a" && names[1] == "b" && names[2] == "c");

        const auto ta = repo.tensor("a");
        assert(ta.dtype == lfm::TensorDType::BF16);
        assert(ta.shape.size() == 1 && ta.shape[0] == 2);
        assert(ta.bytes == 4);
        assert(std::memcmp(ta.data, a_vals, 4) == 0);

        const auto tc = repo.tensor("c");
        assert(tc.shape[0] == 3);
        assert(std::memcmp(tc.data, c_vals, 6) == 0);
    }

    // Single-file backward compatibility via directory with model.safetensors.
    const auto single_dir = std::filesystem::temp_directory_path() / "lfm_repo_single";
    std::filesystem::create_directories(single_dir);
    write_shard(single_dir / "model.safetensors",
                R"({"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",
                a_vals, sizeof(a_vals));
    {
        lfm::SafeTensorRepository repo(single_dir);
        assert(!repo.sharded());
        assert(repo.contains("x"));
        const auto tx = repo.tensor("x");
        assert(tx.shape[0] == 2);
        assert(std::memcmp(tx.data, a_vals, 4) == 0);
    }

    // Missing shard must be detected at construction.
    bool rejected = false;
    try {
        const std::string bad_index =
            R"({"metadata":{},"weight_map":{"z":"model-missing.safetensors"}})";
        std::ofstream bad(dir / "model.safetensors.index.json");
        bad << bad_index;
        lfm::SafeTensorRepository repo(dir);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(single_dir);
    std::cout << "safetensors_repo_test: ok\n";
 } catch (const std::exception& e) {
    std::cerr << "safetensors_repo_test FAILED: " << e.what() << "\n";
    return 1;
 }
}
