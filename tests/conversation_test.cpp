#include "celeg/text/conversation.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    const celeg::ChatMessage assistant{
        celeg::ChatRole::Assistant, std::nullopt,
        {celeg::ToolCall{"call_1", "weather", "{\"city\":\"Paris\"}"}},
        std::nullopt, std::nullopt};
    const celeg::ChatMessage result{
        celeg::ChatRole::Tool, std::string("sunny"), {}, std::string("call_1"), std::nullopt};
    const std::vector<celeg::ChatMessage> valid{assistant, result};
    celeg::validate_conversation(valid);

    bool rejected = false;
    try {
        const std::vector<celeg::ChatMessage> invalid{
            assistant,
            celeg::ChatMessage{celeg::ChatRole::Tool, std::string("bad"), {}, std::string("unknown"), std::nullopt}};
        celeg::validate_conversation(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);

    rejected = false;
    try {
        const std::vector<celeg::ChatMessage> invalid{
            celeg::ChatMessage{celeg::ChatRole::Assistant, std::nullopt,
                               {celeg::ToolCall{"same", "a", "{}"}, celeg::ToolCall{"same", "b", "{}"}},
                               std::nullopt, std::nullopt}};
        celeg::validate_conversation(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);
    std::cout << "conversation_test: ok\n";
    return 0;
}
