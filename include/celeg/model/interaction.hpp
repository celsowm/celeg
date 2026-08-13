#pragma once

#include "celeg/checkpoint/metadata.hpp"

#include <string>

namespace celeg {

// Resolves interaction behavior from the checkpoint's chat-template source.
// The result names a reusable wire protocol; it never names a model family.
// Returns an empty string when the template source doesn't match any
// registered protocol (or when the checkpoint has none) -- resolution
// happens unconditionally at model load, before the caller knows whether a
// chat template will ever be needed (e.g. --raw prompts, --print-config,
// benchmarking tools). ChatTemplateCatalog::find() already throws its own
// "unknown chat template" error at the point a chat-formatted turn actually
// requires one; that is where an unrecognized/empty id should fail loudly.
std::string resolve_chat_template_id(const CheckpointMetadata& metadata);

} // namespace celeg
