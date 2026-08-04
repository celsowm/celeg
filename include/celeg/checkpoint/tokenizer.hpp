#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace celeg {

// Format-neutral tokenizer data extracted while a checkpoint is opened. The
// text tokenizer consumes this normalized payload without knowing whether it
// came from GGUF metadata or another checkpoint format.
struct TokenizerData {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int32_t> token_types;
    int32_t bos_id = 1;
    int32_t eos_id = 7;
    int32_t pad_id = 0;
    std::string pre_tokenizer;
};

} // namespace celeg
