#include "lfm/text/tokenizer.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "lfm_tokenizer_test.json";
    std::ofstream out(path);
    out << R"({
      "model": {
        "type": "BPE",
        "vocab": {"<|startoftext|>":0,"h":1,"i":2,"hi":3,"Ġ":4,"x":5,"'":6,"S":7,"'S":8},
        "merges": ["h i", "' S"]
      },
      "added_tokens": [
        {"id":0,"content":"<|startoftext|>","special":true}
      ]
    })";
    out.close();

    lfm::BpeTokenizer tokenizer(path.string());
    const auto ids = tokenizer.encode("hi", false);
    assert(ids.size() == 1 && ids[0] == 3);
    assert(tokenizer.decode(ids, false) == "hi");
    const auto contraction = tokenizer.encode("'S", false);
    assert(contraction.size() == 1 && contraction[0] == 8);
    std::filesystem::remove(path);
    std::cout << "tokenizer_test: ok\n";
}
