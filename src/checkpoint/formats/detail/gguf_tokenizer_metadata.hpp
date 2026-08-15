#pragma once

#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/tokenizer.hpp"

namespace celeg {

TokenizerData read_gguf_tokenizer_data(const GgufFile& file);

}
