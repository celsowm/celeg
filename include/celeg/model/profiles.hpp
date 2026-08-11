#pragma once

#include "celeg/checkpoint/metadata.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

// A profile is data: it describes a source convention without creating a C++
// type for every revision or parameter count.
struct MetadataPatch {
    std::string key;
    MetadataValue value;
};

struct CheckpointProfile {
    std::string id;
    std::string repository_contains;
    std::vector<MetadataPatch> patches;
    std::string chat_template;
};

class CheckpointProfileResolver {
public:
    static bool matches(const CheckpointProfile& profile,
                        const CheckpointMetadata& metadata) {
        if (profile.repository_contains.empty()) return true;
        std::string haystack = metadata.repository_hint;
        std::string needle = profile.repository_contains;
        std::transform(haystack.begin(), haystack.end(), haystack.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::transform(needle.begin(), needle.end(), needle.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return haystack.find(needle) != std::string::npos;
    }

    static CheckpointMetadata apply(const CheckpointProfile& profile,
                                    const CheckpointMetadata& metadata) {
        CheckpointMetadata patched = metadata;
        for (const MetadataPatch& patch : profile.patches) {
            patched.values[patch.key] = patch.value;
        }
        return patched;
    }
};

} // namespace celeg
