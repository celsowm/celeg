#include "celeg/checkpoint/formats/safetensors.hpp"
#include "checkpoint/detail/binary_codec.hpp"
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
    celeg::binary::write_le(out, header_size);
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
        celeg::SafeTensorFile file(path.string());
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

}

int main() {
    const auto base = std::filesystem::temp_directory_path();
    const auto path = base / "celeg_safetensors_test.safetensors";
    const uint16_t values[2] = {0x3f80, 0x4000};

    write_file(path,
               R"({"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",
               values, sizeof(values));

    {
        celeg::SafeTensorFile file(path.string());
        CELEG_TEST_CHECK(file.contains("x"));
        const auto tensor = file.tensor("x");
        CELEG_TEST_CHECK(tensor.dtype == celeg::TensorDType::BF16);
        CELEG_TEST_CHECK(tensor.shape.size() == 1 && tensor.shape[0] == 2);
        CELEG_TEST_CHECK(tensor.bytes == 4);
        CELEG_TEST_CHECK(std::memcmp(tensor.data, values, 4) == 0);

        celeg::TensorLocator loc = file.locate("x");
        CELEG_TEST_CHECK(loc.bytes == 4);
        CELEG_TEST_CHECK(loc.dtype == celeg::TensorDType::BF16);
        CELEG_TEST_CHECK(loc.shape.size() == 1 && loc.shape[0] == 2);

        std::vector<std::byte> dest(4);
        file.read(loc, dest);
        CELEG_TEST_CHECK(std::memcmp(dest.data(), values, 4) == 0);

        bool throws_size_mismatch = false;
        try {
            std::vector<std::byte> bad_dest(5);
            file.read(loc, bad_dest);
        } catch (const std::invalid_argument&) {
            throws_size_mismatch = true;
        }
        CELEG_TEST_CHECK(throws_size_mismatch);

        bool throws_oob = false;
        try {
            celeg::TensorLocator bad_loc = loc;
            bad_loc.absolute_offset = 100000;
            file.read(bad_loc, dest);
        } catch (const std::out_of_range&) {
            throws_oob = true;
        }
        CELEG_TEST_CHECK(throws_oob);
    }

    CELEG_TEST_CHECK(rejects(path,
                   R"({"x":{"dtype":"BF16","shape":[3],"data_offsets":[0,4]}})",
                   values, sizeof(values)));
    CELEG_TEST_CHECK(rejects(path,
                   R"({"x":{"dtype":"XYZ","shape":[2],"data_offsets":[0,4]}})",
                   values, sizeof(values)));
    CELEG_TEST_CHECK(rejects(path,
                   R"({"x":{"dtype":"BF16","shape":[2],"data_offsets":[-1,3]}})",
                   values, sizeof(values)));
    CELEG_TEST_CHECK(rejects(path,
                   R"({"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"y":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}})",
                   values, sizeof(values)));

    std::filesystem::remove(path);
    std::cout << "safetensors_test: ok\n";
}
