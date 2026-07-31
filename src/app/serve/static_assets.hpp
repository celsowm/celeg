#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace celeg::app::serve {

// Reads a whole file into a string, or nullopt if it can't be opened.
std::optional<std::string> read_file(const std::filesystem::path& path);

// Reads a file that is expected to exist (ships in the source tree / build
// dir under our control). Throws std::runtime_error naming the path if it's
// missing, since that indicates a packaging bug rather than a client error.
std::string read_required_file(const std::filesystem::path& path);

} // namespace celeg::app::serve
