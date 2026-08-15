#pragma once

#include <string>

namespace celeg {

struct ModelGraph;

namespace detail {

std::string program_semantic_fingerprint(const ModelGraph& graph);

}
}
