#pragma once

#include <cstdint>

namespace celeg {

enum class ChatTemplateKind : uint8_t {
    Lfm2Instruct,
    GraniteInstruct,
    Gemma4Instruct,
};

} // namespace celeg
