#include "static_assets.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace celeg::app::serve {

std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string read_required_file(const std::filesystem::path& path) {
    auto contents = read_file(path);
    if (!contents) throw std::runtime_error("required file not found: " + path.string());
    return *contents;
}

}
