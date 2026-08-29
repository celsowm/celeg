#include "celeg/checkpoint/formats/gguf.hpp"

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

namespace celeg {
namespace {

constexpr uint32_t kGgufMagic = 0x46554747u;

class Cursor {
public:
    Cursor(const std::byte* base, size_t size) : base_(base), size_(size) {}

    size_t offset() const { return pos_; }
    void seek(size_t pos) {
        if (pos > size_) throw std::runtime_error("gguf: seek past end");
        pos_ = pos;
    }

    template <typename T>
    T read_scalar() {
        require(sizeof(T));
        T value;
        std::memcpy(&value, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        if constexpr (std::endian::native == std::endian::big) {
            byteswap(&value, sizeof(T));
        }
        return value;
    }

    std::string read_string() {
        const uint64_t len = read_scalar<uint64_t>();
        require(len);
        std::string out(reinterpret_cast<const char*>(base_ + pos_),
                        static_cast<size_t>(len));
        pos_ += static_cast<size_t>(len);
        return out;
    }

private:
    void require(size_t n) const {
        if (pos_ > size_ || n > size_ - pos_) {
            throw std::runtime_error("gguf: truncated file");
        }
    }
    static void byteswap(void* p, size_t n) {
        auto* b = static_cast<uint8_t*>(p);
        for (size_t i = 0; i < n / 2; ++i) std::swap(b[i], b[n - 1 - i]);
    }

    const std::byte* base_;
    size_t size_;
    size_t pos_ = 0;
};

int64_t read_integer_value(Cursor& c, GgufValueKind kind) {
    switch (kind) {
        case GgufValueKind::U8: return c.read_scalar<uint8_t>();
        case GgufValueKind::I8: return c.read_scalar<int8_t>();
        case GgufValueKind::U16: return c.read_scalar<uint16_t>();
        case GgufValueKind::I16: return c.read_scalar<int16_t>();
        case GgufValueKind::U32: return c.read_scalar<uint32_t>();
        case GgufValueKind::I32: return c.read_scalar<int32_t>();
        case GgufValueKind::Bool: return c.read_scalar<uint8_t>() != 0;
        case GgufValueKind::U64:
            return static_cast<int64_t>(c.read_scalar<uint64_t>());
        case GgufValueKind::I64: return c.read_scalar<int64_t>();
        default:
            throw std::runtime_error("gguf: value kind is not integral");
    }
}

bool is_integral_kind(GgufValueKind kind) {
    switch (kind) {
        case GgufValueKind::U8:
        case GgufValueKind::I8:
        case GgufValueKind::U16:
        case GgufValueKind::I16:
        case GgufValueKind::U32:
        case GgufValueKind::I32:
        case GgufValueKind::Bool:
        case GgufValueKind::U64:
        case GgufValueKind::I64:
            return true;
        default:
            return false;
    }
}

GgufValue read_value(Cursor& c, GgufValueKind kind);

GgufValue read_array(Cursor& c) {
    GgufValue value;
    value.kind = GgufValueKind::Array;
    const auto elem_raw = c.read_scalar<uint32_t>();
    value.array_kind = static_cast<GgufValueKind>(elem_raw);
    const uint64_t count = c.read_scalar<uint64_t>();
    if (value.array_kind == GgufValueKind::String) {
        value.array_strings.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            value.array_strings.push_back(c.read_string());
        }
    } else if (value.array_kind == GgufValueKind::F32) {
        value.array_numbers.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            value.array_numbers.push_back(c.read_scalar<float>());
        }
    } else if (value.array_kind == GgufValueKind::F64) {
        value.array_numbers.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            value.array_numbers.push_back(c.read_scalar<double>());
        }
    } else if (is_integral_kind(value.array_kind)) {
        value.array_integers.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            value.array_integers.push_back(read_integer_value(c, value.array_kind));
        }
    } else {
        throw std::runtime_error("gguf: unsupported array element kind");
    }
    return value;
}

GgufValue read_value(Cursor& c, GgufValueKind kind) {
    GgufValue value;
    value.kind = kind;
    switch (kind) {
        case GgufValueKind::String:
            value.str = c.read_string();
            break;
        case GgufValueKind::F32:
            value.number = c.read_scalar<float>();
            break;
        case GgufValueKind::F64:
            value.number = c.read_scalar<double>();
            break;
        case GgufValueKind::Array:
            return read_array(c);
        default:
            value.integer = read_integer_value(c, kind);
            break;
    }
    return value;
}

}


std::vector<int64_t> GgufTensorInfo::hf_shape() const {
    std::vector<int64_t> out;
    out.reserve(dims.size());
    for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        out.push_back(static_cast<int64_t>(*it));
    }
    return out;
}

uint64_t GgufTensorInfo::element_count() const {
    uint64_t count = 1;
    for (uint64_t d : dims) count *= d;
    return count;
}


GgufFile::GgufFile(const std::string& path) {
    try {
#if defined(_WIN32)
        file_handle_ = ::CreateFileA(
            path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            file_handle_ = nullptr;
            throw std::runtime_error("cannot open gguf file: " + path);
        }
        LARGE_INTEGER size{};
        if (::GetFileSizeEx(file_handle_, &size) == 0) {
            throw std::runtime_error("cannot stat gguf file: " + path);
        }
        file_size_ = static_cast<size_t>(size.QuadPart);
        if (file_size_ < 24) throw std::runtime_error("invalid gguf file: too small");
        mapping_handle_ = ::CreateFileMappingA(
            file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle_ == nullptr) {
            throw std::runtime_error("CreateFileMapping failed for gguf file");
        }
        mapping_ = ::MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
        if (mapping_ == nullptr) {
            throw std::runtime_error("MapViewOfFile failed for gguf file");
        }
#else
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("cannot open gguf file: " + path);
        struct stat st {};
        if (::fstat(fd_, &st) != 0) {
            throw std::runtime_error("cannot stat gguf file: " + path);
        }
        file_size_ = static_cast<size_t>(st.st_size);
        if (file_size_ < 24) throw std::runtime_error("invalid gguf file: too small");
        mapping_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            throw std::runtime_error("mmap failed for gguf file");
        }
#endif
        parse();
    } catch (...) {
#if defined(_WIN32)
        if (mapping_) { ::UnmapViewOfFile(mapping_); mapping_ = nullptr; }
        if (mapping_handle_) { ::CloseHandle(mapping_handle_); mapping_handle_ = nullptr; }
        if (file_handle_) { ::CloseHandle(file_handle_); file_handle_ = nullptr; }
#else
        if (mapping_) { ::munmap(mapping_, file_size_); mapping_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
        throw;
    }
}

GgufFile::~GgufFile() {
#if defined(_WIN32)
    if (mapping_) ::UnmapViewOfFile(mapping_);
    if (mapping_handle_) ::CloseHandle(mapping_handle_);
    if (file_handle_) ::CloseHandle(file_handle_);
#else
    if (mapping_) ::munmap(mapping_, file_size_);
    if (fd_ >= 0) ::close(fd_);
#endif
}

void GgufFile::parse() {
    const auto* base = static_cast<const std::byte*>(mapping_);
    Cursor c(base, file_size_);

    const uint32_t magic = c.read_scalar<uint32_t>();
    if (magic != kGgufMagic) throw std::runtime_error("gguf: bad magic");
    version_ = c.read_scalar<uint32_t>();
    if (version_ != 2 && version_ != 3) {
        throw std::runtime_error("gguf: unsupported version " +
                                 std::to_string(version_));
    }
    const uint64_t tensor_count = c.read_scalar<uint64_t>();
    const uint64_t kv_count = c.read_scalar<uint64_t>();

    kv_.reserve(static_cast<size_t>(kv_count));
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = c.read_string();
        const auto kind = static_cast<GgufValueKind>(c.read_scalar<uint32_t>());
        kv_.emplace(std::move(key), read_value(c, kind));
    }

    uint32_t alignment = 32;
    if (const auto it = kv_.find("general.alignment"); it != kv_.end()) {
        if (is_integral_kind(it->second.kind)) {
            alignment = static_cast<uint32_t>(it->second.integer);
        }
    }
    if (alignment == 0) alignment = 32;

    tensors_.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensorInfo info;
        info.name = c.read_string();
        const uint32_t ndim = c.read_scalar<uint32_t>();
        if (ndim > 8) throw std::runtime_error("gguf: too many tensor dims");
        info.dims.reserve(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            info.dims.push_back(c.read_scalar<uint64_t>());
        }
        info.raw_type = c.read_scalar<int32_t>();
        info.type = ggml_type_from_ordinal(info.raw_type);
        info.offset = c.read_scalar<uint64_t>();
        tensors_.emplace(info.name, std::move(info));
    }

    const size_t after_header = c.offset();
    tensor_data_offset_ = (after_header + alignment - 1) / alignment * alignment;
    if (tensor_data_offset_ > file_size_) {
        throw std::runtime_error("gguf: tensor data offset past end");
    }
}

bool GgufFile::has(std::string_view key) const {
    return kv_.find(std::string(key)) != kv_.end();
}

const GgufValue& GgufFile::value(std::string_view key) const {
    const auto it = kv_.find(std::string(key));
    if (it == kv_.end()) {
        throw std::out_of_range("gguf: missing metadata key: " + std::string(key));
    }
    return it->second;
}

uint32_t GgufFile::u32(std::string_view key) const {
    return static_cast<uint32_t>(value(key).integer);
}
uint64_t GgufFile::u64(std::string_view key) const {
    return static_cast<uint64_t>(value(key).integer);
}
int64_t GgufFile::i64(std::string_view key) const { return value(key).integer; }
float GgufFile::f32(std::string_view key) const {
    return static_cast<float>(value(key).number);
}
bool GgufFile::boolean(std::string_view key) const {
    return value(key).integer != 0;
}
const std::string& GgufFile::str(std::string_view key) const {
    return value(key).str;
}

uint32_t GgufFile::u32_or(std::string_view key, uint32_t fallback) const {
    return has(key) ? u32(key) : fallback;
}
float GgufFile::f32_or(std::string_view key, float fallback) const {
    return has(key) ? f32(key) : fallback;
}
bool GgufFile::boolean_or(std::string_view key, bool fallback) const {
    return has(key) ? boolean(key) : fallback;
}
std::string GgufFile::str_or(std::string_view key, const std::string& fallback) const {
    return has(key) ? str(key) : fallback;
}

bool GgufFile::contains_tensor(std::string_view name) const {
    return tensors_.find(std::string(name)) != tensors_.end();
}

const GgufTensorInfo& GgufFile::tensor_info(std::string_view name) const {
    const auto it = tensors_.find(std::string(name));
    if (it == tensors_.end()) {
        throw std::out_of_range("gguf: missing tensor: " + std::string(name));
    }
    return it->second;
}

GgufTensorView GgufFile::tensor(std::string_view name) const {
    const GgufTensorInfo& info = tensor_info(name);
    const GgmlTypeTrait trait = ggml_type_trait(info.type);
    if (trait.block_size == 0) {
        throw std::runtime_error("gguf: unsupported tensor type for " + info.name +
                                 " (" + ggml_type_name(info.type) + ", ordinal " +
                                 std::to_string(info.raw_type) + ")");
    }
    const uint64_t elems = info.element_count();
    if (elems % static_cast<uint64_t>(trait.block_size) != 0) {
        throw std::runtime_error("gguf: element count not block-aligned for " +
                                 info.name);
    }
    const uint64_t blocks = elems / static_cast<uint64_t>(trait.block_size);
    const size_t bytes = static_cast<size_t>(blocks) *
                         static_cast<size_t>(trait.type_size);
    const size_t abs = tensor_data_offset_ + static_cast<size_t>(info.offset);
    if (abs > file_size_ || bytes > file_size_ - abs) {
        throw std::runtime_error("gguf: tensor payload out of range for " + info.name);
    }
    GgufTensorView view;
    view.type = info.type;
    view.shape = info.hf_shape();
    view.data = static_cast<const std::byte*>(mapping_) + abs;
    view.bytes = bytes;
    view.element_count = elems;
    return view;
}

std::vector<std::string> GgufFile::tensor_names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) out.push_back(name);
    return out;
}

}
