#pragma once

#include "celeg/model/inference.hpp"

#include "../support.hpp"

#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

/// Generic checkpoint-metadata accessors shared by the inventory schedule,
/// norm-layout and rope phases. Moved verbatim from `inventory.cpp`; kept
/// separate from `metadata/aliases.hpp` on purpose: the conflict/coercion
/// semantics here differ, so merging them would be a false DRY.
AttentionPatternKind parse_attention_pattern(std::string_view value, std::string_view source);
const MetadataValue* metadata_alias(const CheckpointMetadata& metadata, std::string_view key);
std::optional<int> integer_alias(const CheckpointMetadata& metadata, std::string_view key);
std::optional<double> numeric_alias(const CheckpointMetadata& metadata, std::string_view key);
std::optional<std::string> string_alias(const CheckpointMetadata& metadata, std::string_view key);
std::optional<std::string> string_aliases( const CheckpointMetadata& metadata, std::initializer_list<std::string_view> keys, std::string_view fact);
std::optional<double> numeric_aliases( const CheckpointMetadata& metadata, std::initializer_list<std::string_view> keys, std::string_view fact);
std::optional<int> integer_aliases( const CheckpointMetadata& metadata, std::initializer_list<std::string_view> keys, std::string_view fact);

}
