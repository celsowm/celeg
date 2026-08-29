// Generates tests/data/gguf_iq_reference.inc: real IQ-quantized blocks lifted
// from a cached GGUF file, paired with the float values upstream ggml decodes
// them to.
//
// The IQ formats are codebook-indexed, so a hand-derived expectation would
// only restate whatever the decoder under test already does. Taking the
// reference from ggml itself makes the fixture authoritative.
//
// This tool is deliberately outside the CMake build: it links against the
// vendored llama.cpp in .externals/, which is a developer checkout rather
// than a dependency of the engine. Regenerate with:
//
//   c++ -std=c++20 -O2 tests/tools/gen_gguf_iq_reference.cpp \
//       -I .externals/llama.cpp/ggml/include \
//       -L .externals/llama.cpp/build-cpu/bin -lggml-base \
//       -Wl,-rpath,$PWD/.externals/llama.cpp/build-cpu/bin \
//       -o /tmp/gen_iq && /tmp/gen_iq <model.gguf> > tests/data/gguf_iq_reference.inc

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The five IQ types celeg decodes, with the ggml enum they correspond to.
struct IqType {
    const char* name;
    ggml_type type;
    int block_size;
    int type_size;
};

constexpr IqType kIqTypes[] = {
    {"IQ2_S", GGML_TYPE_IQ2_S, 256, 82},
    {"IQ3_XXS", GGML_TYPE_IQ3_XXS, 256, 98},
    {"IQ3_S", GGML_TYPE_IQ3_S, 256, 110},
    {"IQ4_NL", GGML_TYPE_IQ4_NL, 32, 18},
    {"IQ4_XS", GGML_TYPE_IQ4_XS, 256, 136},
};

/// How many consecutive elements of each type to capture: four 256-wide rows,
/// which is enough to exercise every sub-block scale path and lets the test
/// feed the fixture straight into a 256-column GgmlMatrixView regardless of
/// whether the type packs 32 or 256 elements per block.
constexpr int kElementsPerType = 1024;

constexpr int blocks_for(const IqType& iq) { return kElementsPerType / iq.block_size; }

struct Reader {
    std::ifstream in;
    explicit Reader(const std::string& path) : in(path, std::ios::binary) {
        if (!in) throw std::runtime_error("cannot open " + path);
    }
    template <typename T> T scalar() {
        T value{};
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!in) throw std::runtime_error("short read");
        return value;
    }
    std::string text() {
        const auto length = scalar<uint64_t>();
        std::string value(length, '\0');
        in.read(value.data(), static_cast<std::streamsize>(length));
        return value;
    }
    void skip_value(uint32_t kind) {
        switch (kind) {
            case 8: text(); return;
            case 9: {
                const auto inner = scalar<uint32_t>();
                const auto count = scalar<uint64_t>();
                for (uint64_t i = 0; i < count; ++i) skip_value(inner);
                return;
            }
            default: {
                static const std::map<uint32_t, int> sizes{
                    {0, 1}, {1, 1}, {2, 2}, {3, 2}, {4, 4}, {5, 4},
                    {6, 4}, {7, 1}, {10, 8}, {11, 8}, {12, 8}};
                in.seekg(sizes.at(kind), std::ios::cur);
            }
        }
    }
};

struct TensorEntry {
    std::string name;
    int32_t type = 0;
    uint64_t elements = 1;
    uint64_t offset = 0;
};

void emit(const IqType& iq, const std::vector<uint8_t>& blocks) {
    const int elements = kElementsPerType;
    std::vector<float> reference(static_cast<size_t>(elements));
    const ggml_type_traits* traits = ggml_get_type_traits(iq.type);
    if (!traits || !traits->to_float) {
        throw std::runtime_error(std::string("ggml has no to_float for ") + iq.name);
    }
    traits->to_float(blocks.data(), reference.data(), elements);

    std::printf("// %s: %d blocks of %d elements\n", iq.name, blocks_for(iq),
                iq.block_size);
    std::printf("constexpr unsigned char k%sBlocks[] = {", iq.name);
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (i % 16 == 0) std::printf("\n    ");
        std::printf("0x%02x,", blocks[i]);
    }
    std::printf("\n};\n");
    std::printf("constexpr float k%sReference[] = {", iq.name);
    for (size_t i = 0; i < reference.size(); ++i) {
        if (i % 6 == 0) std::printf("\n    ");
        std::printf("%.9gf,", static_cast<double>(reference[i]));
    }
    std::printf("\n};\n\n");
}

// Captures whichever of the wanted types `path` happens to contain. No
// single quant recipe carries all five, so the caller passes several files
// and each contributes what it has.
void harvest(const std::string& path, std::vector<bool>& done) {
    Reader reader(path);

    if (reader.scalar<uint32_t>() != 0x46554747u) {
        throw std::runtime_error("not a GGUF file: " + path);
    }
    reader.scalar<uint32_t>();  // version
    const auto tensor_count = reader.scalar<uint64_t>();
    const auto kv_count = reader.scalar<uint64_t>();
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < kv_count; ++i) {
        const std::string key = reader.text();
        const auto kind = reader.scalar<uint32_t>();
        if (key == "general.alignment" && kind == 4) {
            alignment = reader.scalar<uint32_t>();
        } else {
            reader.skip_value(kind);
        }
    }

    std::vector<TensorEntry> tensors;
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorEntry entry;
        entry.name = reader.text();
        const auto dims = reader.scalar<uint32_t>();
        for (uint32_t d = 0; d < dims; ++d) entry.elements *= reader.scalar<uint64_t>();
        entry.type = reader.scalar<int32_t>();
        entry.offset = reader.scalar<uint64_t>();
        tensors.push_back(std::move(entry));
    }
    const auto header_end = static_cast<uint64_t>(reader.in.tellg());
    const uint64_t data_start = (header_end + alignment - 1) / alignment * alignment;

    for (size_t index = 0; index < std::size(kIqTypes); ++index) {
        const IqType& iq = kIqTypes[index];
        if (done[index]) continue;
        const TensorEntry* found = nullptr;
        for (const TensorEntry& entry : tensors) {
            if (entry.type == static_cast<int32_t>(iq.type) &&
                entry.elements >= static_cast<uint64_t>(kElementsPerType)) {
                found = &entry;
                break;
            }
        }
        if (!found) continue;
        const size_t bytes = static_cast<size_t>(blocks_for(iq)) * iq.type_size;
        std::vector<uint8_t> blocks(bytes);
        reader.in.seekg(static_cast<std::streamoff>(data_start + found->offset));
        reader.in.read(reinterpret_cast<char*>(blocks.data()),
                       static_cast<std::streamsize>(bytes));
        if (!reader.in) throw std::runtime_error("short tensor read for " + found->name);
        std::fprintf(stderr, "%-8s <- %s :: %s\n", iq.name,
                     path.substr(path.find_last_of('/') + 1).c_str(), found->name.c_str());
        emit(iq, blocks);
        done[index] = true;
    }
}

}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [more.gguf ...]\n", argv[0]);
        return 2;
    }
    std::printf("// Generated by tests/tools/gen_gguf_iq_reference.cpp.\n");
    std::printf("// Blocks are lifted verbatim from cached GGUF files; the reference\n");
    std::printf("// values come from ggml's own to_float. Do not hand-edit.\n\n");

    std::vector<bool> done(std::size(kIqTypes), false);
    for (int i = 1; i < argc; ++i) harvest(argv[i], done);

    int missing = 0;
    for (size_t i = 0; i < done.size(); ++i) {
        if (!done[i]) {
            std::fprintf(stderr, "no %s tensor in any input file\n", kIqTypes[i].name);
            ++missing;
        }
    }
    return missing == 0 ? 0 : 1;
}
