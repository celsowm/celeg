#include "lfm/checkpoint/formats/safetensors.hpp"
#include "lfm/detail/binary_codec.hpp"
#include "support/assertions.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_file(const std::filesystem::path& path,
                const std::string& header,
                const void* data,
                size_t data_size) {
    const uint64_t header_size = header.size();
    std::ofstream out(path, std::ios::binary);
    lfm::binary::write_le(out, header_size);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (data_size) {
        out.write(static_cast<const char*>(data),
                  static_cast<std::streamsize>(data_size));
    }
}

bool rejects(const std::filesystem::path& path,
             const std::string& header,
             const void* data,
             size_t data_size) {
    write_file(path, header, data, data_size);
    try {
        lfm::SafeTensorFile file(path.string());
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

} // namespace

int main() {
    const auto base = std::filesystem::temp_directory_path();
    const auto path = base / "lfm_safetensors_test.safetensors";
    const uint16_t values[2] = {0x3f80, 0x4000};

    write_file(path,
               R"({"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",
               values, sizeof(values));

    {
        lfm::SafeTensorFile file(path.string());
        LFM_TEST_CHECK(file.contains("x"));
        const auto tensor = file.tensor("x");
        LFM_TEST_CHECK(tensor.dtype == lfm::TensorDType::BF16);
        LFM_TEST_CHECK(tensor.shape.size() == 1 && tensor.shape[0] == 2);
        LFM_TEST_CHECK(tensor.bytes == 4);
        LFM_TEST_CHECK(std::memcmp(tensor.data, values, 4) == 0);

        // Test locate and read
        lfm::TensorLocator loc = file.locate("x");
        LFM_TEST_CHECK(loc.bytes == 4);
        LFM_TEST_CHECK(loc.dtype == lfm::TensorDType::BF16);
        LFM_TEST_CHECK(loc.shape.size() == 1 && loc.shape[0] == 2);

        std::vector<std::byte> dest(4);
        file.read(loc, dest);
        LFM_TEST_CHECK(std::memcmp(dest.data(), values, 4) == 0);

        // Destination size mismatch throws exception
        bool throws_size_mismatch = false;
        try {
            std::vector<std::byte> bad_dest(5);
            file.read(loc, bad_dest);
        } catch (const std::invalid_argument&) {
            throws_size_mismatch = true;
        }
        LFM_TEST_CHECK(throws_size_mismatch);

        // Out of bounds locator throws exception
        bool throws_oob = false;
        try {
            lfm::TensorLocator bad_loc = loc;
            bad_loc.absolute_offset = 100000;
            file.read(bad_loc, dest);
        } catch (const std::out_of_range&) {
            throws_oob = true;
        }
        LFM_TEST_CHECK(throws_oob);
    }

    LFM_TEST_CHECK(rejects(path,
                   R"({"x":{"dtype":"BF16","shape":[3],"data_offsets":[0,4]}})",
                   values, sizeof(values)));
    LFM_TEST_CHECK(rejects(path,
                   R"({"x":{"dtype":"XYZ","shape":[2],"data_offsets":[0,4]}})",
                   values, sizeof(values)));
    LFM_TEST_CHECK(rejects(path,
                   R"({"x":{"dtype":"BF16","shape":[2],"data_offsets":[-1,3]}})",
                   values, sizeof(values)));
    LFM_TEST_CHECK(rejects(path,
                   R"({"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"y":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}})",
                   values, sizeof(values)));

    std::filesystem::remove(path);
    std::cout << "safetensors_test: ok\n";
}
