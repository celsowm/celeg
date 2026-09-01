#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace celeg::test_support {

struct SyntheticTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<std::uint16_t> values;
};

inline std::uint16_t bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x8000u) >> 16);
}

inline std::size_t element_count(const std::vector<int>& shape) {
    std::size_t count = 1;
    for (const int dimension : shape) {
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

inline SyntheticTensor pattern_tensor(std::string name, std::vector<int> shape,
                                      float base, float step) {
    SyntheticTensor tensor{std::move(name), std::move(shape), {}};
    tensor.values.resize(element_count(tensor.shape));
    for (std::size_t index = 0; index < tensor.values.size(); ++index) {
        tensor.values[index] = bf16(
            base + static_cast<float>((index * 7 + 3) % 17) * step);
    }
    return tensor;
}

inline SyntheticTensor constant_tensor(std::string name, std::vector<int> shape,
                                       float value) {
    SyntheticTensor tensor{std::move(name), std::move(shape), {}};
    tensor.values.assign(element_count(tensor.shape), bf16(value));
    return tensor;
}

inline void write_safetensors_checkpoint(
    const std::filesystem::path& directory,
    std::string_view model_type,
    const std::vector<SyntheticTensor>& tensors) {
    std::filesystem::create_directories(directory);
    std::ofstream config(directory / "config.json");
    config << "{\"model_type\":\"" << model_type << "\"}";

    std::ostringstream header;
    header << '{';
    std::size_t offset = 0;
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        if (index != 0) header << ',';
        const SyntheticTensor& tensor = tensors[index];
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
    header << '}';

    const std::string header_text = header.str();
    std::ofstream weights(directory / "model.safetensors", std::ios::binary);
    const std::uint64_t header_size = static_cast<std::uint64_t>(header_text.size());
    weights.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    weights.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const SyntheticTensor& tensor : tensors) {
        weights.write(reinterpret_cast<const char*>(tensor.values.data()),
                      static_cast<std::streamsize>(
                          tensor.values.size() * sizeof(std::uint16_t)));
    }
}

}  // namespace celeg::test_support
