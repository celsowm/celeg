#include "celeg/text/tokenizer.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

std::string make_prompt(size_t target_bytes) {
    static constexpr const char* corpus =
        "The quick brown fox jumps over 1234567890.\n"
        "Unicode: café Ελληνικά 日本語 한국어 العربية.\n"
        "Code: for (size_t i = 0; i < 1024; ++i) value += i;\n";
    std::string result;
    result.reserve(target_bytes);
    while (result.size() < target_bytes) result += corpus;
    result.resize(target_bytes);
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: tokenizer_benchmark <tokenizer.json> [bytes]\n";
        return 2;
    }
    const size_t bytes = argc == 3 ? std::stoull(argv[2]) : 4096;
    celeg::BpeTokenizer tokenizer(std::filesystem::path(argv[1]).string());
    const std::string prompt = make_prompt(bytes);

    constexpr int warmup = 3;
    constexpr int iterations = 20;
    for (int i = 0; i < warmup; ++i) tokenizer.encode(prompt, false);

    size_t token_count = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        token_count += tokenizer.encode(prompt, false).size();
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const double seconds_per_iteration = elapsed / iterations;
    const double chars_per_second = static_cast<double>(bytes) / seconds_per_iteration;
    std::cout << "bytes=" << bytes
              << " tokens=" << (token_count / iterations)
              << " seconds_per_encode=" << seconds_per_iteration
              << " chars_per_second=" << chars_per_second << '\n';
    return 0;
}
