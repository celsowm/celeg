#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace celeg::tokenizer_detail {

void append_utf8(std::string& out, std::uint32_t code_point);
std::pair<std::uint32_t, std::size_t> next_cp(
    std::string_view text, std::size_t offset);
bool is_space_cp(std::uint32_t code_point);
bool is_number_cp(std::uint32_t code_point);
bool is_punctuation_or_symbol_cp(std::uint32_t code_point);
int category(std::uint32_t code_point);
bool ascii_case_equal(
    std::string_view text, std::size_t offset, std::string_view expected);
std::string pair_key(const std::string& left, const std::string& right);

}
