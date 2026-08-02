#include "celeg/checkpoint/catalog.hpp"

#include "celeg/checkpoint/formats/json.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/repositories/gguf.hpp"
#include "celeg/checkpoint/repositories/safetensors.hpp"

#include <algorithm>
#include <string>
#include <stdexcept>

namespace celeg {
namespace {

std::filesystem::path config_path_for(const std::filesystem::path& path) {
    const auto root = std::filesystem::is_directory(path)
        ? path : path.parent_path();
    return root / "config.json";
}

class GgufFormat final : public ICheckpointFormat {
public:
    std::string_view id() const override { return "gguf"; }

    bool matches(const std::filesystem::path& path) const override {
        return !std::filesystem::is_directory(path) &&
               path.extension() == ".gguf";
    }

    CheckpointView open(const std::filesystem::path& path) const override {
        auto file = std::make_shared<GgufFile>(path.string());
        CheckpointView result;
        result.path = path;
        result.metadata = CheckpointMetadata::from_gguf(*file);
        result.repository = std::make_shared<GgufRepository>(std::move(file));
        return result;
    }
};

class SafeTensorsFormat final : public ICheckpointFormat {
public:
    std::string_view id() const override { return "safetensors"; }

    bool matches(const std::filesystem::path& path) const override {
        if (std::filesystem::is_regular_file(path)) {
            return path.extension() == ".safetensors";
        }
        if (!std::filesystem::is_directory(path)) return false;
        return std::filesystem::exists(path / "model.safetensors") ||
               std::filesystem::exists(path / "model.safetensors.index.json");
    }

    CheckpointView open(const std::filesystem::path& path) const override {
        const auto config = config_path_for(path);
        if (!std::filesystem::exists(config)) {
            throw std::runtime_error(
                "config.json not found alongside checkpoint: " + config.string());
        }
        CheckpointView result;
        result.path = path;
        result.metadata = CheckpointMetadata::from_json(Json::parse_file(config.string()));
        result.repository = std::make_shared<SafeTensorRepository>(path);
        return result;
    }
};

} // namespace

void CheckpointFormatCatalog::add(std::unique_ptr<ICheckpointFormat> format) {
    if (frozen_) throw std::logic_error("checkpoint format catalog is frozen");
    if (!format || format->id().empty()) {
        throw std::invalid_argument("checkpoint format must have a non-empty id");
    }
    const auto duplicate = std::find_if(formats_.begin(), formats_.end(),
        [&](const auto& existing) { return existing->id() == format->id(); });
    if (duplicate != formats_.end()) {
        throw std::invalid_argument("duplicate checkpoint format id: " +
                                    std::string(format->id()));
    }
    formats_.push_back(std::move(format));
}

void CheckpointFormatCatalog::freeze() { frozen_ = true; }

const ICheckpointFormat& CheckpointFormatCatalog::select(
    const std::filesystem::path& path) const {
    const ICheckpointFormat* selected = nullptr;
    for (const auto& format : formats_) {
        if (!format->matches(path)) continue;
        if (selected != nullptr) {
            throw std::runtime_error("multiple checkpoint formats match: " +
                                     path.string());
        }
        selected = format.get();
    }
    if (selected == nullptr) {
        throw std::runtime_error("no registered checkpoint format matches: " +
                                 path.string());
    }
    return *selected;
}

CheckpointView CheckpointFormatCatalog::open(
    const std::filesystem::path& path) const {
    return select(path).open(path);
}

std::vector<std::string_view> CheckpointFormatCatalog::ids() const {
    std::vector<std::string_view> result;
    result.reserve(formats_.size());
    for (const auto& format : formats_) result.push_back(format->id());
    return result;
}

std::shared_ptr<const CheckpointFormatCatalog>
create_builtin_checkpoint_format_catalog() {
    auto catalog = std::make_shared<CheckpointFormatCatalog>();
    catalog->add(std::make_unique<GgufFormat>());
    catalog->add(std::make_unique<SafeTensorsFormat>());
    catalog->freeze();
    return catalog;
}

} // namespace celeg
