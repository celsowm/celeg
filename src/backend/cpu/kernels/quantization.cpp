#include "lfm/backend/cpu/quantization.hpp"
#include "lfm/model/weights/quantization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace lfm {
namespace {

constexpr std::array<char, 8> kMagic{{'L','F','M','P','A','C','K','1'}};
constexpr uint32_t kKindQ4 = 1;
constexpr uint32_t kKindBf16 = 2;

size_t checked_mul(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string(label) + " overflow");
    }
    return a * b;
}

int decode_q4(uint8_t nibble) {
    nibble &= 0x0fU;
    return nibble >= 8 ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

uint8_t encode_q4(int value) {
    return static_cast<uint8_t>(value) & 0x0fU;
}

template <typename Read>
Q4GroupMatrix quantize_impl(size_t rows, size_t cols, size_t group_size, Read read) {
    if (rows == 0 || cols == 0) throw std::invalid_argument("Q4 matrix dimensions must be positive");
    if (group_size == 0 || group_size > 256 || (group_size % 2) != 0) {
        throw std::invalid_argument("Q4 group size must be even and between 2 and 256");
    }
    Q4GroupMatrix result;
    result.rows = static_cast<uint32_t>(rows);
    result.cols = static_cast<uint32_t>(cols);
    result.group_size = static_cast<uint32_t>(group_size);
    result.groups_per_row = static_cast<uint32_t>((cols + group_size - 1) / group_size);
    result.values.assign(checked_mul(rows, (cols + 1) / 2, "Q4 values"), 0);
    result.scales_bf16.resize(checked_mul(rows, result.groups_per_row, "Q4 scales"));

    const size_t row_bytes = (cols + 1) / 2;
    for (size_t row = 0; row < rows; ++row) {
        uint8_t* packed = result.values.data() + row * row_bytes;
        for (size_t group = 0; group < result.groups_per_row; ++group) {
            const size_t begin = group * group_size;
            const size_t end = std::min(cols, begin + group_size);
            float maximum = 0.0f;
            for (size_t col = begin; col < end; ++col) {
                const float value = read(row, col);
                if (!std::isfinite(value)) throw std::invalid_argument("Q4 source contains non-finite value");
                maximum = std::max(maximum, std::abs(value));
            }
            const float scale = maximum > 0.0f ? maximum / 7.0f : 1.0f;
            result.scales_bf16[row * result.groups_per_row + group] =
                float_to_bf16_bits(scale);
            for (size_t col = begin; col < end; ++col) {
                const int q = std::clamp(static_cast<int>(std::nearbyint(read(row, col) / scale)), -7, 7);
                uint8_t& byte = packed[col / 2];
                const uint8_t nibble = encode_q4(q);
                if ((col & 1U) == 0) byte = static_cast<uint8_t>((byte & 0xf0U) | nibble);
                else byte = static_cast<uint8_t>((byte & 0x0fU) | (nibble << 4));
            }
        }
    }
    result.validate();
    return result;
}

struct PackHeader {
    std::array<char, 8> magic{};
    uint32_t version = 0;
    uint32_t entries = 0;
    uint32_t group_size = 0;
    uint32_t reserved = 0;
    uint64_t source_len = 0;
    uint64_t isa_len = 0;
};

struct EntryHeader {
    uint32_t kind = 0;
    uint32_t name_len = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t group_size = 0;
    uint32_t groups_per_row = 0;
    uint64_t values_bytes = 0;
    uint64_t scales_bytes = 0;
};

void write_exact(std::ofstream& out, const void* data, size_t bytes) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("failed writing CPU pack");
}

void read_exact(std::ifstream& in, void* data, size_t bytes) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("truncated CPU pack");
}

} // namespace

void Q4GroupMatrix::validate() const {
    if (rows == 0 || cols == 0 || group_size == 0 || groups_per_row == 0) {
        throw std::invalid_argument("invalid Q4 matrix metadata");
    }
    const size_t expected_values = static_cast<size_t>(rows) * packed_values_per_row();
    const size_t expected_scales = static_cast<size_t>(rows) * groups_per_row;
    if (values.size() != expected_values || scales_bf16.size() != expected_scales) {
        throw std::invalid_argument("invalid Q4 matrix storage size");
    }
}

Q4GroupMatrix quantize_bf16_groupwise_q4(const std::byte* data,
                                          size_t rows, size_t cols,
                                          size_t group_size) {
    if (!data) throw std::invalid_argument("null BF16 Q4 source");
    return quantize_impl(rows, cols, group_size,
        [data, cols](size_t row, size_t col) {
            uint16_t bits = 0;
            std::memcpy(&bits, data + (row * cols + col) * sizeof(uint16_t), sizeof(bits));
            return bf16_bits_to_float(bits);
        });
}

Q4GroupMatrix quantize_float_groupwise_q4(const float* data,
                                           size_t rows, size_t cols,
                                           size_t group_size) {
    if (!data) throw std::invalid_argument("null float Q4 source");
    return quantize_impl(rows, cols, group_size,
        [data, cols](size_t row, size_t col) { return data[row * cols + col]; });
}

void dequantize_q4_row(const Q4GroupMatrix& matrix, size_t row, float* output) {
    matrix.validate();
    if (!output || row >= matrix.rows) throw std::invalid_argument("invalid Q4 row request");
    const uint8_t* values = matrix.values.data() + row * matrix.packed_values_per_row();
    for (size_t col = 0; col < matrix.cols; ++col) {
        const uint8_t byte = values[col / 2];
        const uint8_t nibble = (col & 1U) == 0 ? byte & 0x0fU : byte >> 4;
        const size_t group = col / matrix.group_size;
        const float scale = bf16_bits_to_float(
            matrix.scales_bf16[row * matrix.groups_per_row + group]);
        output[col] = static_cast<float>(decode_q4(nibble)) * scale;
    }
}


void Q8GroupVector::validate() const {
    if (elements == 0 || group_size == 0 || groups == 0) {
        throw std::runtime_error("invalid Q8 activation dimensions");
    }
    const size_t expected_groups =
        (static_cast<size_t>(elements) + group_size - 1) / group_size;
    if (groups != expected_groups || values.size() != elements ||
        scales.size() != groups || sums.size() != groups) {
        throw std::runtime_error("invalid Q8 activation storage");
    }
    for (float scale : scales) {
        if (!std::isfinite(scale) || scale < 0.0f) {
            throw std::runtime_error("invalid Q8 activation scale");
        }
    }
}

Q8GroupVector quantize_float_groupwise_q8(const float* data,
                                           size_t elements,
                                           size_t group_size) {
    if (!data || elements == 0 || group_size == 0 ||
        elements > std::numeric_limits<uint32_t>::max() ||
        group_size > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("invalid Q8 activation arguments");
    }
    Q8GroupVector result;
    result.elements = static_cast<uint32_t>(elements);
    result.group_size = static_cast<uint32_t>(group_size);
    result.groups = static_cast<uint32_t>((elements + group_size - 1) / group_size);
    result.values.resize(elements);
    result.scales.resize(result.groups);
    result.sums.resize(result.groups);
    for (size_t group = 0; group < result.groups; ++group) {
        const size_t begin = group * group_size;
        const size_t end = std::min(elements, begin + group_size);
        float maximum = 0.0f;
        for (size_t i = begin; i < end; ++i) {
            if (!std::isfinite(data[i])) {
                throw std::invalid_argument("Q8 activation contains a non-finite value");
            }
            maximum = std::max(maximum, std::abs(data[i]));
        }
        const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
        result.scales[group] = scale;
        if (scale == 0.0f) {
            std::fill(result.values.begin() + static_cast<ptrdiff_t>(begin),
                      result.values.begin() + static_cast<ptrdiff_t>(end), int8_t{0});
            result.sums[group] = 0;
            continue;
        }
        const float inverse = 1.0f / scale;
        int32_t sum = 0;
        for (size_t i = begin; i < end; ++i) {
            const long quantized = std::lround(data[i] * inverse);
            result.values[i] = static_cast<int8_t>(std::clamp<long>(quantized, -127, 127));
            sum += result.values[i];
        }
        result.sums[group] = sum;
    }
    result.validate();
    return result;
}

struct CpuPackWriter::Impl {
    std::filesystem::path path;
    CpuPackMetadata metadata;
    struct Entry {
        std::string name;
        uint32_t kind = 0;
        Q4GroupMatrix q4;
        std::vector<uint16_t> bf16;
    };
    std::vector<Entry> entries;
    bool committed = false;
};

CpuPackWriter::CpuPackWriter(const std::filesystem::path& path, CpuPackMetadata metadata)
    : impl_(new Impl{path, std::move(metadata), {}, false}) {}
CpuPackWriter::~CpuPackWriter() { delete impl_; }

void CpuPackWriter::add_q4_matrix(const std::string& name, const Q4GroupMatrix& matrix) {
    if (impl_->committed || name.empty()) throw std::logic_error("invalid CPU pack writer state");
    matrix.validate();
    Impl::Entry entry;
    entry.name = name;
    entry.kind = kKindQ4;
    entry.q4 = matrix;
    impl_->entries.push_back(std::move(entry));
}

void CpuPackWriter::add_bf16_vector(const std::string& name,
                                    const std::byte* data, size_t elements) {
    if (impl_->committed || name.empty() || (!data && elements != 0)) {
        throw std::logic_error("invalid CPU pack vector");
    }
    Impl::Entry entry;
    entry.name = name;
    entry.kind = kKindBf16;
    entry.bf16.resize(elements);
    if (elements) std::memcpy(entry.bf16.data(), data, elements * sizeof(uint16_t));
    impl_->entries.push_back(std::move(entry));
}

void CpuPackWriter::commit() {
    if (impl_->committed) throw std::logic_error("CPU pack already committed");
    const auto temporary = impl_->path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create CPU pack: " + temporary);
    PackHeader header;
    header.magic = kMagic;
    header.version = impl_->metadata.version;
    header.entries = static_cast<uint32_t>(impl_->entries.size());
    header.group_size = impl_->metadata.group_size;
    header.source_len = impl_->metadata.source_id.size();
    header.isa_len = impl_->metadata.isa.size();
    write_exact(out, &header, sizeof(header));
    write_exact(out, impl_->metadata.source_id.data(), impl_->metadata.source_id.size());
    write_exact(out, impl_->metadata.isa.data(), impl_->metadata.isa.size());
    for (const Impl::Entry& entry : impl_->entries) {
        EntryHeader eh;
        eh.kind = entry.kind;
        eh.name_len = static_cast<uint32_t>(entry.name.size());
        if (entry.kind == kKindQ4) {
            eh.rows = entry.q4.rows; eh.cols = entry.q4.cols;
            eh.group_size = entry.q4.group_size;
            eh.groups_per_row = entry.q4.groups_per_row;
            eh.values_bytes = entry.q4.values.size();
            eh.scales_bytes = entry.q4.scales_bf16.size() * sizeof(uint16_t);
        } else {
            eh.rows = 1; eh.cols = static_cast<uint32_t>(entry.bf16.size());
            eh.values_bytes = entry.bf16.size() * sizeof(uint16_t);
        }
        write_exact(out, &eh, sizeof(eh));
        write_exact(out, entry.name.data(), entry.name.size());
        if (entry.kind == kKindQ4) {
            write_exact(out, entry.q4.values.data(), entry.q4.values.size());
            write_exact(out, entry.q4.scales_bf16.data(), eh.scales_bytes);
        } else {
            write_exact(out, entry.bf16.data(), eh.values_bytes);
        }
    }
    out.flush();
    if (!out) throw std::runtime_error("failed finalizing CPU pack");
    out.close();
    std::filesystem::rename(temporary, impl_->path);
    impl_->committed = true;
}

struct CpuPackReader::Impl {
    CpuPackMetadata metadata;
    struct Entry {
        uint32_t kind = 0;
        Q4GroupMatrix q4;
        std::vector<uint16_t> bf16;
    };
    std::unordered_map<std::string, Entry> entries;
};

CpuPackReader::CpuPackReader(const std::filesystem::path& path) : impl_(new Impl) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open CPU pack: " + path.string());
    PackHeader header;
    read_exact(in, &header, sizeof(header));
    if (header.magic != kMagic || header.version != 1 ||
        header.source_len > (1u << 20) || header.isa_len > 128 ||
        header.entries > 10000) {
        throw std::runtime_error("invalid CPU pack header");
    }
    impl_->metadata.version = header.version;
    impl_->metadata.group_size = header.group_size;
    impl_->metadata.source_id.resize(static_cast<size_t>(header.source_len));
    impl_->metadata.isa.resize(static_cast<size_t>(header.isa_len));
    read_exact(in, impl_->metadata.source_id.data(), impl_->metadata.source_id.size());
    read_exact(in, impl_->metadata.isa.data(), impl_->metadata.isa.size());
    for (uint32_t i = 0; i < header.entries; ++i) {
        EntryHeader eh;
        read_exact(in, &eh, sizeof(eh));
        if (eh.name_len == 0 || eh.name_len > 4096 || eh.values_bytes > (1ull << 34) ||
            eh.scales_bytes > (1ull << 32)) {
            throw std::runtime_error("invalid CPU pack entry header");
        }
        std::string name(eh.name_len, '\0');
        read_exact(in, name.data(), name.size());
        Impl::Entry entry;
        entry.kind = eh.kind;
        if (eh.kind == kKindQ4) {
            entry.q4.rows = eh.rows; entry.q4.cols = eh.cols;
            entry.q4.group_size = eh.group_size;
            entry.q4.groups_per_row = eh.groups_per_row;
            entry.q4.values.resize(static_cast<size_t>(eh.values_bytes));
            entry.q4.scales_bf16.resize(static_cast<size_t>(eh.scales_bytes / sizeof(uint16_t)));
            read_exact(in, entry.q4.values.data(), entry.q4.values.size());
            read_exact(in, entry.q4.scales_bf16.data(), static_cast<size_t>(eh.scales_bytes));
            entry.q4.validate();
        } else if (eh.kind == kKindBf16) {
            if ((eh.values_bytes % sizeof(uint16_t)) != 0) throw std::runtime_error("invalid BF16 CPU pack entry");
            entry.bf16.resize(static_cast<size_t>(eh.values_bytes / sizeof(uint16_t)));
            read_exact(in, entry.bf16.data(), static_cast<size_t>(eh.values_bytes));
        } else {
            throw std::runtime_error("unknown CPU pack entry kind");
        }
        if (!impl_->entries.emplace(std::move(name), std::move(entry)).second) {
            throw std::runtime_error("duplicate CPU pack entry");
        }
    }
}
CpuPackReader::~CpuPackReader() { delete impl_; }
const CpuPackMetadata& CpuPackReader::metadata() const { return impl_->metadata; }
bool CpuPackReader::contains(const std::string& name) const { return impl_->entries.contains(name); }
Q4GroupMatrix CpuPackReader::read_q4_matrix(const std::string& name) const {
    const auto it = impl_->entries.find(name);
    if (it == impl_->entries.end() || it->second.kind != kKindQ4) throw std::runtime_error("missing Q4 CPU pack entry: " + name);
    return it->second.q4;
}
std::vector<float> CpuPackReader::read_bf16_vector(const std::string& name) const {
    const auto it = impl_->entries.find(name);
    if (it == impl_->entries.end() || it->second.kind != kKindBf16) throw std::runtime_error("missing BF16 CPU pack entry: " + name);
    std::vector<float> result(it->second.bf16.size());
    for (size_t i = 0; i < result.size(); ++i) result[i] = bf16_bits_to_float(it->second.bf16[i]);
    return result;
}

} // namespace lfm
