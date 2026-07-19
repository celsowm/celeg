#include "lfm/chat_template.hpp"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    auto tmpl = lfm::make_chat_template(lfm::ChatTemplateKind::Lfm2Instruct);
    assert(tmpl);
    assert(tmpl->kind() == lfm::ChatTemplateKind::Lfm2Instruct);

    const std::string user_only = tmpl->format("Hello", "");
    assert(user_only == "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");

    const std::string with_system = tmpl->format("Hi", "You are helpful.");
    assert(with_system ==
        "<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");

    std::cout << "chat_template_test: ok\n";
}
