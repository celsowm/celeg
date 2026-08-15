#include "celeg/serve/protocol/openapi.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#ifndef CELEG_OPENAPI_SPEC_PATH
#error "CELEG_OPENAPI_SPEC_PATH must be defined by the build (see CMakeLists.txt)"
#endif

namespace celeg::serve::protocol {

namespace {

std::string read_spec_template() {
    std::ifstream in(CELEG_OPENAPI_SPEC_PATH, std::ios::binary);
    if (!in) throw std::runtime_error("openapi.json not found at " CELEG_OPENAPI_SPEC_PATH);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

}

std::string build_openapi_spec(const std::string& model_name) {
    std::string spec = read_spec_template();
    std::size_t pos = 0;
    while ((pos = spec.find("__MODEL_NAME__", pos)) != std::string::npos) {
        spec.replace(pos, std::string("__MODEL_NAME__").size(), model_name);
        pos += model_name.size();
    }
    return spec;
}

}
