#pragma once

#include "celeg/checkpoint/view.hpp"

#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace celeg {

class ICheckpointFormat {
public:
    virtual ~ICheckpointFormat() = default;
    virtual std::string_view id() const = 0;
    virtual bool matches(const std::filesystem::path& path) const = 0;
    virtual CheckpointView open(const std::filesystem::path& path) const = 0;
};

class CheckpointFormatCatalog {
public:
    void add(std::unique_ptr<ICheckpointFormat> format);
    void freeze();
    const ICheckpointFormat& select(const std::filesystem::path& path) const;
    CheckpointView open(const std::filesystem::path& path) const;
    std::vector<std::string_view> ids() const;

private:
    bool frozen_ = false;
    std::vector<std::unique_ptr<ICheckpointFormat>> formats_;
};

std::shared_ptr<const CheckpointFormatCatalog>
create_builtin_checkpoint_format_catalog();
void add_builtin_checkpoint_formats(CheckpointFormatCatalog& catalog);

} // namespace celeg
