#include "celeg/text/tokenizer.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: tokenizer_probe <tokenizer.json> <text>\n";
        return 2;
    }
    celeg::BpeTokenizer tokenizer(argv[1]);
    std::string text = argv[2];
    if (text.size() > 1 && text[0] == '@') {
        std::ifstream input(text.substr(1), std::ios::binary);
        std::ostringstream contents;
        contents << input.rdbuf();
        text = contents.str();
    }
    const auto ids = tokenizer.encode(text, false);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << ids[i];
    }
    std::cout << " decoded=" << tokenizer.decode(ids, false) << '\n';
    return 0;
}
