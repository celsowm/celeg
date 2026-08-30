#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace celeg::app::serve {

std::optional<std::string> read_file(const std::filesystem::path& path);

std::string read_required_file(const std::filesystem::path& path);

}
