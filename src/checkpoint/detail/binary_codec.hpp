#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace celeg::binary {

template <typename T, bool = std::is_enum_v<T>>
struct storage_type { using type = T; };

template <typename T>
struct storage_type<T, true> { using type = std::underlying_type_t<T>; };

template <typename T>
requires(std::is_integral_v<T> || std::is_enum_v<T>)
void write_le(std::ostream& output, T value) {
    using U = std::make_unsigned_t<typename storage_type<T>::type>;
    std::array<char, sizeof(T)> bytes{};
    U bits = static_cast<U>(value);
    for (size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<char>((bits >> (index * 8)) & 0xffU);
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("binary write failed");
}

template <typename T>
requires(std::is_integral_v<T> || std::is_enum_v<T>)
T read_le(std::istream& input) {
    using U = std::make_unsigned_t<typename storage_type<T>::type>;
    std::array<char, sizeof(T)> bytes{};
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("binary read failed");
    U bits = 0;
    for (size_t index = 0; index < bytes.size(); ++index) {
        bits |= static_cast<U>(static_cast<unsigned char>(bytes[index])) << (index * 8);
    }
    return static_cast<T>(bits);
}

inline void write_bytes(std::ostream& output, std::span<const std::byte> bytes) {
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("binary write failed");
}

template <typename T>
requires(std::is_integral_v<T> || std::is_enum_v<T>)
T read_le(std::span<const std::byte> bytes, size_t& offset) {
    using U = std::make_unsigned_t<typename storage_type<T>::type>;
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T)) {
        throw std::out_of_range("binary input is truncated");
    }
    U bits = 0;
    for (size_t index = 0; index < sizeof(T); ++index) {
        bits |= static_cast<U>(std::to_integer<unsigned char>(bytes[offset + index]))
            << (index * 8);
    }
    offset += sizeof(T);
    return static_cast<T>(bits);
}

}
