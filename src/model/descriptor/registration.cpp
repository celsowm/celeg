#include "celeg/model/descriptor.hpp"
#include "detail.hpp"

#include "celeg/checkpoint/formats/json.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace celeg {

void register_descriptor_architectures(ArchitectureCatalog& catalog,
                                       const std::filesystem::path& directory) {
    // Descriptor files are an optional extension layer.  The built-in
    // automatic architecture is registered independently, so a descriptorless
    // installation must remain usable after all declarative stores are removed.
    if (!std::filesystem::is_directory(directory)) return;

    std::vector<std::filesystem::path> stores;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            stores.push_back(entry.path());
        }
    }
    std::sort(stores.begin(), stores.end());
    if (stores.empty()) return;

    for (const std::filesystem::path& file : stores) {
        const Json root = Json::parse_file(file.string());
        for (const Json& descriptor :
             descriptor_detail::required(root, "architectures").as_array()) {
            const std::string descriptor_id =
                descriptor.is_object() && descriptor.contains("id")
                    ? descriptor.at("id").as_string() : "<unknown>";
            try {
                catalog.add(descriptor_detail::make_descriptor_architecture(
                    descriptor_detail::parse_descriptor(descriptor)));
            } catch (const std::exception& error) {
                throw std::invalid_argument(
                    "invalid model descriptor " + descriptor_id + " in " +
                    file.filename().string() + ": " + error.what());
            }
        }
    }
}

} // namespace celeg
