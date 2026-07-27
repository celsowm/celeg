#include "lfm/text/chat_template.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <string>

int main() {
    auto tmpl = lfm::make_chat_template(lfm::ChatTemplateKind::Lfm2Instruct);
    LFM_TEST_CHECK(tmpl);
    LFM_TEST_CHECK(tmpl->kind() == lfm::ChatTemplateKind::Lfm2Instruct);

    const std::string user_only = tmpl->format("Hello", "");
    LFM_TEST_CHECK(user_only == "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");

    const std::string with_system = tmpl->format("Hi", "You are helpful.");
    LFM_TEST_CHECK(with_system ==
        "<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");

    std::cout << "chat_template_test: ok\n";
}
