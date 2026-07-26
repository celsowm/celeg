#include "lfm/safetensors.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fileapi.h>
#include <memoryapi.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lfm {
namespace {

TensorDType parse_dtype(const std::string& name) {
    if (name == "BF16") return TensorDType::BF16;
    if (name == "F16") return TensorDType::F16;
    if (name == "F32") return TensorDType::F32;
    if (name == "I8") return TensorDType::I8;
    return TensorDType::Unknown;
}

size_t dtype_size(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::BF16:
        case TensorDType::F16:
            return 2;
        case TensorDType::F32:
            return 4;
        case TensorDType::I8:
            return 1;
        case TensorDType::Unknown:
            return 0;
    }
    return 0;
}

uint64_t read_u64_le(const std::byte* data) {
    uint64_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    if constexpr (std::endian::native == std::endian::big) {
#if defined(_MSC_VER)
        return _byteswap_uint64(value);
#else
        return __builtin_bswap64(value);
#endif
    }
    return value;
}

size_t checked_element_count(const std::vector<int64_t>& shape,
                             const std::string& name) {
    size_t result = 1;
    for (int64_t dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("negative dimension for tensor: " + name);
        }
        if (dimension == 0) return 0;
        const size_t value = static_cast<size_t>(dimension);
        if (result > std::numeric_limits<size_t>::max() / value) {
            throw std::runtime_error("shape overflow for tensor: " + name);
        }
        result *= value;
    }
    return result;
}

} // namespace

SafeTensorFile::SafeTensorFile(const std::string& path) {
    try {
#if defined(_WIN32)
        file_handle_ = ::CreateFileA(
            path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            file_handle_ = nullptr;
            throw std::runtime_error("cannot open safetensors file: " + path);
        }

        LARGE_INTEGER size{};
        if (::GetFileSizeEx(file_handle_, &size) == 0) {
            throw std::runtime_error("cannot stat safetensors file: " + path);
        }
        if (size.QuadPart < 0 ||
            static_cast<uint64_t>(size.QuadPart) >
                std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("unsupported safetensors file size");
        }
        file_size_ = static_cast<size_t>(size.QuadPart);
        if (file_size_ < 8) {
            throw std::runtime_error("invalid safetensors file: too small");
        }

        mapping_handle_ = ::CreateFileMappingA(
            file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle_ == nullptr) {
            throw std::runtime_error("CreateFileMapping failed for safetensors file");
        }

        mapping_ = ::MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
        if (mapping_ == nullptr) {
            throw std::runtime_error("MapViewOfFile failed for safetensors file");
        }
#else
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open safetensors file: " + path);
        }

        struct stat st {};
        if (::fstat(fd_, &st) != 0) {
            throw std::runtime_error("cannot stat safetensors file: " + path);
        }
        if (st.st_size < 0 ||
            static_cast<uint64_t>(st.st_size) >
                std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("unsupported safetensors file size");
        }
        file_size_ = static_cast<size_t>(st.st_size);
        if (file_size_ < 8) {
            throw std::runtime_error("invalid safetensors file: too small");
        }

        mapping_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            throw std::runtime_error("mmap failed for safetensors file");
        }
#endif

        const auto* bytes = static_cast<const std::byte*>(mapping_);
        const uint64_t header_size_u64 = read_u64_le(bytes);
        if (header_size_u64 > file_size_ - 8 ||
            header_size_u64 > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("invalid safetensors header length");
        }
        const size_t header_size = static_cast<size_t>(header_size_u64);
        data_offset_ = 8 + header_size;

        const std::string_view header(
            reinterpret_cast<const char*>(bytes + 8), header_size);
        const Json root = Json::parse(header);
        if (!root.is_object()) {
            throw std::runtime_error("safetensors header is not an object");
        }

        struct Range {
            size_t begin;
            size_t end;
            std::string name;
        };
        std::vector<Range> ranges;

        for (const auto& [name, descriptor] : root.as_object()) {
            if (name == "__metadata__") continue;
            if (!descriptor.is_object()) {
                throw std::runtime_error("invalid descriptor for tensor: " + name);
            }

            const auto& offsets = descriptor["data_offsets"].as_array();
            if (offsets.size() != 2) {
                throw std::runtime_error("invalid offsets for tensor: " + name);
            }
            const int64_t begin_i64 = offsets[0].as_i64();
            const int64_t end_i64 = offsets[1].as_i64();
            if (begin_i64 < 0 || end_i64 < 0) {
                throw std::runtime_error("negative offset for tensor: " + name);
            }

            Entry entry;
            entry.dtype = parse_dtype(descriptor["dtype"].as_string());
            if (entry.dtype == TensorDType::Unknown) {
                throw std::runtime_error("unsupported dtype for tensor: " + name);
            }
            for (const Json& dimension : descriptor["shape"].as_array()) {
                entry.shape.push_back(dimension.as_i64());
            }
            entry.begin = static_cast<size_t>(begin_i64);
            entry.end = static_cast<size_t>(end_i64);

            if (entry.begin > entry.end ||
                entry.end > file_size_ - data_offset_) {
                throw std::runtime_error("out-of-range tensor: " + name);
            }

            const size_t element_count = checked_element_count(entry.shape, name);
            const size_t element_size = dtype_size(entry.dtype);
            if (element_count >
                std::numeric_limits<size_t>::max() / element_size) {
                throw std::runtime_error("byte-size overflow for tensor: " + name);
            }
            const size_t expected_bytes = element_count * element_size;
            if (entry.end - entry.begin != expected_bytes) {
                throw std::runtime_error("byte count does not match shape for tensor: " + name);
            }

            auto [_, inserted] = entries_.emplace(name, entry);
            if (!inserted) {
                throw std::runtime_error("duplicate tensor name: " + name);
            }
            ranges.push_back({entry.begin, entry.end, name});
        }

        std::sort(ranges.begin(), ranges.end(),
                  [](const Range& left, const Range& right) {
                      if (left.begin != right.begin) return left.begin < right.begin;
                      return left.end < right.end;
                  });
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].begin < ranges[i - 1].end) {
                throw std::runtime_error(
                    "overlapping tensor ranges: " + ranges[i - 1].name +
                    " and " + ranges[i].name);
            }
        }
    } catch (...) {
#if defined(_WIN32)
        if (mapping_) {
            ::UnmapViewOfFile(mapping_);
            mapping_ = nullptr;
        }
        if (mapping_handle_) {
            ::CloseHandle(mapping_handle_);
            mapping_handle_ = nullptr;
        }
        if (file_handle_) {
            ::CloseHandle(file_handle_);
            file_handle_ = nullptr;
        }
#else
        if (mapping_) {
            ::munmap(mapping_, file_size_);
            mapping_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        throw;
    }
}

SafeTensorFile::~SafeTensorFile() {
#if defined(_WIN32)
    if (mapping_) ::UnmapViewOfFile(mapping_);
    if (mapping_handle_) ::CloseHandle(mapping_handle_);
    if (file_handle_) ::CloseHandle(file_handle_);
#else
    if (mapping_) ::munmap(mapping_, file_size_);
    if (fd_ >= 0) ::close(fd_);
#endif
}

bool SafeTensorFile::contains(std::string_view name) const {
    return entries_.find(std::string(name)) != entries_.end();
}

HostTensorView SafeTensorFile::tensor(std::string_view name) const {
    const auto it = entries_.find(std::string(name));
    if (it == entries_.end()) {
        throw std::out_of_range("missing tensor: " + std::string(name));
    }
    const Entry& entry = it->second;
    const auto* base =
        static_cast<const std::byte*>(mapping_) + data_offset_;
    return HostTensorView{
        entry.dtype,
        entry.shape,
        base + entry.begin,
        entry.end - entry.begin,
    };
}

std::vector<std::string> SafeTensorFile::names() const {
    std::vector<std::string> result;
    result.reserve(entries_.size());
    for (const auto& [name, _] : entries_) result.push_back(name);
    std::sort(result.begin(), result.end());
    return result;
}

TensorLocator SafeTensorFile::locate(std::string_view name, std::uint32_t shard_id) const {
    const auto it = entries_.find(std::string(name));
    if (it == entries_.end()) {
        throw std::out_of_range("missing tensor: " + std::string(name));
    }
    const Entry& entry = it->second;
    TensorLocator locator;
    locator.shard_id = shard_id;
    locator.absolute_offset = data_offset_ + entry.begin;
    locator.bytes = entry.end - entry.begin;
    locator.dtype = entry.dtype;
    locator.shape = entry.shape;
    return locator;
}

void SafeTensorFile::read(const TensorLocator& locator, std::span<std::byte> destination) const {
    if (destination.size() != locator.bytes) {
        throw std::invalid_argument("read: destination span size (" +
                                    std::to_string(destination.size()) +
                                    ") does not match requested bytes (" +
                                    std::to_string(locator.bytes) + ")");
    }
    if (locator.absolute_offset + locator.bytes > file_size_) {
        throw std::out_of_range("read: locator range [" +
                                std::to_string(locator.absolute_offset) + ", " +
                                std::to_string(locator.absolute_offset + locator.bytes) +
                                "] exceeds file size " + std::to_string(file_size_));
    }
    if (locator.bytes == 0) return;

#if defined(_WIN32)
    // Windows positioned read using ReadFile with OVERLAPPED
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(locator.absolute_offset & 0xFFFFFFFF);
    overlapped.OffsetHigh = static_cast<DWORD>((locator.absolute_offset >> 32) & 0xFFFFFFFF);

    std::byte* dest_ptr = destination.data();
    size_t bytes_to_read = locator.bytes;
    while (bytes_to_read > 0) {
        DWORD chunk = static_cast<DWORD>(std::min<size_t>(bytes_to_read, 0x7FFFFFFF));
        DWORD bytes_read = 0;
        if (!::ReadFile(file_handle_, dest_ptr, chunk, &bytes_read, &overlapped)) {
            DWORD err = ::GetLastError();
            if (err != ERROR_IO_PENDING) {
                throw std::runtime_error("Windows ReadFile failed with error: " + std::to_string(err));
            }
        }
        if (bytes_read == 0) {
            throw std::runtime_error("Windows ReadFile returned 0 bytes (unexpected EOF)");
        }
        dest_ptr += bytes_read;
        bytes_to_read -= bytes_read;

        // Update offset in overlapped for next iteration
        std::uint64_t next_offset = (static_cast<std::uint64_t>(overlapped.OffsetHigh) << 32) | overlapped.Offset;
        next_offset += bytes_read;
        overlapped.Offset = static_cast<DWORD>(next_offset & 0xFFFFFFFF);
        overlapped.OffsetHigh = static_cast<DWORD>((next_offset >> 32) & 0xFFFFFFFF);
    }
#else
    // POSIX positioned read using pread
    std::byte* dest_ptr = destination.data();
    size_t bytes_to_read = locator.bytes;
    off_t offset = static_cast<off_t>(locator.absolute_offset);
    while (bytes_to_read > 0) {
        ssize_t n = ::pread(fd_, dest_ptr, bytes_to_read, offset);
        if (n < 0) {
            if (errno == EINTR) continue; // retry interrupted reads
            throw std::runtime_error("POSIX pread failed: " + std::string(std::strerror(errno)));
        }
        if (n == 0) {
            throw std::runtime_error("POSIX pread returned 0 bytes (unexpected EOF)");
        }
        dest_ptr += n;
        bytes_to_read -= n;
        offset += n;
    }
#endif
}

} // namespace lfm
