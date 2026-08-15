#include "celeg/backend/cpu/quantization.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unordered_map>

namespace celeg {
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

uint64_t checked_add(uint64_t a, uint64_t b, const char* label) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        throw std::overflow_error(std::string(label) + " overflow");
    }
    return a + b;
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
    if (bytes == 0) return;
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("failed writing CPU pack");
}

void read_exact(std::ifstream& in, void* data, size_t bytes) {
    if (bytes == 0) return;
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("truncated CPU pack");
}

uint64_t stream_position(std::istream& in) {
    const std::streampos position = in.tellg();
    if (position < 0) throw std::runtime_error("invalid CPU pack stream position");
    return static_cast<uint64_t>(position);
}

void seek_to(std::ifstream& in, uint64_t offset) {
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("CPU pack offset exceeds stream limits");
    }
    in.clear();
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) throw std::runtime_error("cannot seek CPU pack");
}

std::ifstream open_pack(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open CPU pack: " + path.string());
    return in;
}

PackHeader make_header(const CpuPackMetadata& metadata, uint32_t entries) {
    PackHeader header;
    header.magic = kMagic;
    header.version = metadata.version;
    header.entries = entries;
    header.group_size = metadata.group_size;
    header.source_len = metadata.source_id.size();
    header.isa_len = metadata.isa.size();
    return header;
}

}

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

void quantize_float_groupwise_q8_into(const float* data,
                                       size_t elements,
                                       size_t group_size,
                                       int8_t* values,
                                       float* scales,
                                       int32_t* sums) {
    if (!data || !values || !scales || !sums || elements == 0 || group_size == 0) {
        throw std::invalid_argument("invalid Q8 activation arguments");
    }
    const size_t groups = (elements + group_size - 1) / group_size;
    for (size_t group = 0; group < groups; ++group) {
        const size_t begin = group * group_size;
        const size_t end = std::min(elements, begin + group_size);
        float maximum = 0.0f;
        for (size_t i = begin; i < end; ++i) {
            const float value = std::abs(data[i]);
            if (!(value <= maximum)) maximum = value;
        }
        if (!std::isfinite(maximum)) {
            throw std::invalid_argument("Q8 activation contains a non-finite value");
        }
        const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
        scales[group] = scale;
        if (scale == 0.0f) {
            std::fill(values + begin, values + end, int8_t{0});
            sums[group] = 0;
            continue;
        }
        const float inverse = 1.0f / scale;
        int32_t sum = 0;
        for (size_t i = begin; i < end; ++i) {
            const int quantized = static_cast<int>(std::nearbyintf(data[i] * inverse));
            values[i] = static_cast<int8_t>(quantized);
            sum += quantized;
        }
        sums[group] = sum;
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
    quantize_float_groupwise_q8_into(data, elements, group_size, result.values.data(),
                                     result.scales.data(), result.sums.data());
    result.validate();
    return result;
}

struct CpuPackWriterState {
    std::filesystem::path path;
    std::filesystem::path temporary;
    CpuPackMetadata metadata;
    std::ofstream out;
    uint32_t entries = 0;
    bool committed = false;
};

CpuPackWriter::CpuPackWriter(const std::filesystem::path& path,
                             CpuPackMetadata metadata)
    : state_(std::make_unique<CpuPackWriterState>()) {
    if (path.empty() || metadata.source_id.size() > (1u << 20) ||
        metadata.isa.size() > 128) {
        throw std::invalid_argument("invalid CPU pack metadata");
    }
    state_->path = path;
    state_->temporary = path.string() + ".tmp";
    state_->metadata = std::move(metadata);
    state_->out.open(state_->temporary, std::ios::binary | std::ios::trunc);
    if (!state_->out) {
        throw std::runtime_error("cannot create CPU pack: " +
                                 state_->temporary.string());
    }
    const PackHeader header = make_header(state_->metadata, 0);
    write_exact(state_->out, &header, sizeof(header));
    write_exact(state_->out, state_->metadata.source_id.data(),
                state_->metadata.source_id.size());
    write_exact(state_->out, state_->metadata.isa.data(),
                state_->metadata.isa.size());
}

CpuPackWriter::~CpuPackWriter() {
    if (!state_) return;
    if (state_->out.is_open()) state_->out.close();
    if (!state_->committed) {
        std::error_code error;
        std::filesystem::remove(state_->temporary, error);
    }
}

void CpuPackWriter::add_q4_matrix(const std::string& name,
                                  const Q4GroupMatrix& matrix) {
    if (state_->committed || !state_->out || name.empty() || name.size() > 4096 ||
        state_->entries == std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("invalid CPU pack writer state");
    }
    matrix.validate();
    EntryHeader header;
    header.kind = kKindQ4;
    header.name_len = static_cast<uint32_t>(name.size());
    header.rows = matrix.rows;
    header.cols = matrix.cols;
    header.group_size = matrix.group_size;
    header.groups_per_row = matrix.groups_per_row;
    header.values_bytes = matrix.values.size();
    header.scales_bytes = matrix.scales_bf16.size() * sizeof(uint16_t);
    write_exact(state_->out, &header, sizeof(header));
    write_exact(state_->out, name.data(), name.size());
    write_exact(state_->out, matrix.values.data(), matrix.values.size());
    write_exact(state_->out, matrix.scales_bf16.data(),
                static_cast<size_t>(header.scales_bytes));
    ++state_->entries;
}

void CpuPackWriter::add_bf16_vector(const std::string& name,
                                    const std::byte* data, size_t elements) {
    if (state_->committed || !state_->out || name.empty() || name.size() > 4096 ||
        (!data && elements != 0) ||
        elements > std::numeric_limits<uint32_t>::max() ||
        state_->entries == std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("invalid CPU pack vector");
    }
    EntryHeader header;
    header.kind = kKindBf16;
    header.name_len = static_cast<uint32_t>(name.size());
    header.rows = 1;
    header.cols = static_cast<uint32_t>(elements);
    header.values_bytes = elements * sizeof(uint16_t);
    write_exact(state_->out, &header, sizeof(header));
    write_exact(state_->out, name.data(), name.size());
    write_exact(state_->out, data, static_cast<size_t>(header.values_bytes));
    ++state_->entries;
}

void CpuPackWriter::commit() {
    if (state_->committed || !state_->out) {
        throw std::logic_error("CPU pack already committed");
    }
    state_->out.flush();
    if (!state_->out) throw std::runtime_error("failed finalizing CPU pack");
    state_->out.seekp(0, std::ios::beg);
    if (!state_->out) throw std::runtime_error("failed seeking CPU pack header");
    const PackHeader header = make_header(state_->metadata, state_->entries);
    write_exact(state_->out, &header, sizeof(header));
    state_->out.flush();
    if (!state_->out) throw std::runtime_error("failed finalizing CPU pack header");
    state_->out.close();

    std::error_code error;
    std::filesystem::rename(state_->temporary, state_->path, error);
    if (error) {
        std::error_code remove_error;
        std::filesystem::remove(state_->path, remove_error);
        error.clear();
        std::filesystem::rename(state_->temporary, state_->path, error);
    }
    if (error) {
        throw std::runtime_error("cannot publish CPU pack: " + error.message());
    }
    state_->committed = true;
}

struct CpuPackReaderState {
    struct Entry {
        uint32_t kind = 0;
        uint32_t rows = 0;
        uint32_t cols = 0;
        uint32_t group_size = 0;
        uint32_t groups_per_row = 0;
        uint64_t values_offset = 0;
        uint64_t values_bytes = 0;
        uint64_t scales_offset = 0;
        uint64_t scales_bytes = 0;
    };

    std::filesystem::path path;
    CpuPackMetadata metadata;
    std::unordered_map<std::string, Entry> entries;
};

CpuPackReader::CpuPackReader(const std::filesystem::path& path)
    : state_(std::make_unique<CpuPackReaderState>()) {
    state_->path = path;
    std::ifstream in = open_pack(path);
    PackHeader header;
    read_exact(in, &header, sizeof(header));
    if (header.magic != kMagic || header.version != 2 ||
        header.source_len > (1u << 20) || header.isa_len > 128 ||
        header.entries > 10000) {
        throw std::runtime_error("invalid CPU pack header");
    }
    state_->metadata.version = header.version;
    state_->metadata.group_size = header.group_size;
    state_->metadata.source_id.resize(static_cast<size_t>(header.source_len));
    state_->metadata.isa.resize(static_cast<size_t>(header.isa_len));
    read_exact(in, state_->metadata.source_id.data(), state_->metadata.source_id.size());
    read_exact(in, state_->metadata.isa.data(), state_->metadata.isa.size());

    const uint64_t file_bytes = std::filesystem::file_size(path);
    for (uint32_t index = 0; index < header.entries; ++index) {
        EntryHeader encoded;
        read_exact(in, &encoded, sizeof(encoded));
        if (encoded.name_len == 0 || encoded.name_len > 4096 ||
            encoded.values_bytes > (1ull << 34) ||
            encoded.scales_bytes > (1ull << 32)) {
            throw std::runtime_error("invalid CPU pack entry header");
        }
        std::string name(encoded.name_len, '\0');
        read_exact(in, name.data(), name.size());

        CpuPackReaderState::Entry entry;
        entry.kind = encoded.kind;
        entry.rows = encoded.rows;
        entry.cols = encoded.cols;
        entry.group_size = encoded.group_size;
        entry.groups_per_row = encoded.groups_per_row;
        entry.values_offset = stream_position(in);
        entry.values_bytes = encoded.values_bytes;
        entry.scales_offset = checked_add(entry.values_offset, entry.values_bytes,
                                          "CPU pack scale offset");
        entry.scales_bytes = encoded.scales_bytes;
        const uint64_t end = checked_add(entry.scales_offset, entry.scales_bytes,
                                         "CPU pack entry end");
        if (end > file_bytes) throw std::runtime_error("truncated CPU pack entry");

        if (entry.kind == kKindQ4) {
            if (entry.rows == 0 || entry.cols == 0 || entry.group_size == 0 ||
                entry.groups_per_row !=
                    (entry.cols + entry.group_size - 1) / entry.group_size) {
                throw std::runtime_error("invalid Q4 CPU pack metadata");
            }
            const uint64_t expected_values =
                static_cast<uint64_t>(entry.rows) * ((entry.cols + 1ULL) / 2ULL);
            const uint64_t expected_scales =
                static_cast<uint64_t>(entry.rows) * entry.groups_per_row *
                sizeof(uint16_t);
            if (entry.values_bytes != expected_values ||
                entry.scales_bytes != expected_scales) {
                throw std::runtime_error("invalid Q4 CPU pack storage size");
            }
        } else if (entry.kind == kKindBf16) {
            if (entry.scales_bytes != 0 ||
                (entry.values_bytes % sizeof(uint16_t)) != 0 ||
                entry.values_bytes / sizeof(uint16_t) != entry.cols) {
                throw std::runtime_error("invalid BF16 CPU pack entry");
            }
        } else {
            throw std::runtime_error("unknown CPU pack entry kind");
        }

        if (!state_->entries.emplace(std::move(name), entry).second) {
            throw std::runtime_error("duplicate CPU pack entry");
        }
        seek_to(in, end);
    }
}

CpuPackReader::~CpuPackReader() = default;

const CpuPackMetadata& CpuPackReader::metadata() const {
    return state_->metadata;
}

bool CpuPackReader::contains(const std::string& name) const {
    return state_->entries.contains(name);
}

Q4GroupMatrix CpuPackReader::read_q4_matrix(const std::string& name) const {
    const auto iterator = state_->entries.find(name);
    if (iterator == state_->entries.end() || iterator->second.kind != kKindQ4) {
        throw std::runtime_error("missing Q4 CPU pack entry: " + name);
    }
    const CpuPackReaderState::Entry& entry = iterator->second;
    Q4GroupMatrix matrix;
    matrix.rows = entry.rows;
    matrix.cols = entry.cols;
    matrix.group_size = entry.group_size;
    matrix.groups_per_row = entry.groups_per_row;
    matrix.values.resize(static_cast<size_t>(entry.values_bytes));
    matrix.scales_bf16.resize(
        static_cast<size_t>(entry.scales_bytes / sizeof(uint16_t)));

    std::ifstream in = open_pack(state_->path);
    seek_to(in, entry.values_offset);
    read_exact(in, matrix.values.data(), matrix.values.size());
    seek_to(in, entry.scales_offset);
    read_exact(in, matrix.scales_bf16.data(),
               static_cast<size_t>(entry.scales_bytes));
    matrix.validate();
    return matrix;
}

std::vector<float> CpuPackReader::read_bf16_vector(const std::string& name) const {
    const auto iterator = state_->entries.find(name);
    if (iterator == state_->entries.end() || iterator->second.kind != kKindBf16) {
        throw std::runtime_error("missing BF16 CPU pack entry: " + name);
    }
    const CpuPackReaderState::Entry& entry = iterator->second;
    std::vector<uint16_t> bits(
        static_cast<size_t>(entry.values_bytes / sizeof(uint16_t)));
    std::ifstream in = open_pack(state_->path);
    seek_to(in, entry.values_offset);
    read_exact(in, bits.data(), static_cast<size_t>(entry.values_bytes));

    std::vector<float> result(bits.size());
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = bf16_bits_to_float(bits[index]);
    }
    return result;
}

}
