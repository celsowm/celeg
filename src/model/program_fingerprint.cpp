#include "program_fingerprint.hpp"

#include "celeg/model/graph.hpp"

#include <cstdint>
#include <sstream>

namespace celeg::detail {

std::string program_semantic_fingerprint(const ModelGraph& graph) {
    const std::string semantics = graph.fingerprint();
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : semantics) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

}
