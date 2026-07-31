#include "celeg/text/chat_template.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    auto tmpl = celeg::make_chat_template(celeg::ChatTemplateKind::Lfm2Instruct);
    CELEG_TEST_CHECK(tmpl);
    CELEG_TEST_CHECK(tmpl->kind() == celeg::ChatTemplateKind::Lfm2Instruct);

    const std::string user_only = tmpl->format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Hello"}}, true);
    CELEG_TEST_CHECK(user_only == "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");

    const std::string with_system = tmpl->format(
        std::vector<celeg::ChatMessage>{
            {celeg::ChatRole::System, "You are helpful."},
            {celeg::ChatRole::User, "Hi"},
        },
        true);
    CELEG_TEST_CHECK(with_system ==
        "<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");

    const std::string multi_turn = tmpl->format(
        std::vector<celeg::ChatMessage>{
            {celeg::ChatRole::System, "You are helpful."},
            {celeg::ChatRole::User, "Hi"},
            {celeg::ChatRole::Assistant, "Hello!"},
            {celeg::ChatRole::User, "How are you?"},
        },
        true);
    CELEG_TEST_CHECK(multi_turn ==
        "<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nHello!<|im_end|>\n"
        "<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n");

    const std::string no_generation_prompt = tmpl->format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Hi"}}, false);
    CELEG_TEST_CHECK(no_generation_prompt ==
        "<|startoftext|><|im_start|>user\nHi<|im_end|>\n");

    bool threw = false;
    try {
        (void)tmpl->format(
            std::vector<celeg::ChatMessage>{{celeg::ChatRole::Tool, "result"}}, true);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CELEG_TEST_CHECK(threw);

    std::cout << "chat_template_test: ok\n";
}
