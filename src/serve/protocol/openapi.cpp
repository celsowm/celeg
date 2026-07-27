#include "lfm/serve/protocol/openapi.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

// Points at the source-tree openapi.json (set via CMake target_compile_definitions
// on lfm_serve_protocol), so the spec ships as a real JSON file editable/lintable
// on its own rather than as a C++ string literal.
#ifndef LFM_OPENAPI_SPEC_PATH
#error "LFM_OPENAPI_SPEC_PATH must be defined by the build (see CMakeLists.txt)"
#endif

namespace lfm::serve::protocol {

namespace {

std::string read_spec_template() {
    std::ifstream in(LFM_OPENAPI_SPEC_PATH, std::ios::binary);
    if (!in) throw std::runtime_error("openapi.json not found at " LFM_OPENAPI_SPEC_PATH);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

} // namespace

std::string build_openapi_spec(const std::string& model_name) {
    // __MODEL_NAME__ is substituted here so the example payloads reflect the
    // model this instance actually serves.
    std::string spec = read_spec_template();
    std::size_t pos = 0;
    while ((pos = spec.find("__MODEL_NAME__", pos)) != std::string::npos) {
        spec.replace(pos, std::string("__MODEL_NAME__").size(), model_name);
        pos += model_name.size();
    }
    return spec;
}

} // namespace lfm::serve::protocol
