#pragma once

#include "celeg/checkpoint/metadata.hpp"

#include <string>

namespace celeg {

// Resolves interaction behavior from the checkpoint's chat-template source.
// The result names a reusable wire protocol; it never names a model family.
std::string resolve_chat_template_id(const CheckpointMetadata& metadata);

} // namespace celeg
