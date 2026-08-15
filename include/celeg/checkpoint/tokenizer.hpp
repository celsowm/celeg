#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace celeg {

struct TokenizerData {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int32_t> token_types;
    int32_t bos_id = 1;
    int32_t eos_id = 7;
    int32_t pad_id = 0;
    std::string pre_tokenizer;
};

}
