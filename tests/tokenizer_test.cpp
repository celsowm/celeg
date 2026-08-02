#include "celeg/text/tokenizer.hpp"
#include "support/assertions.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "celeg_tokenizer_test.json";
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

    celeg::BpeTokenizer tokenizer(path.string());
    const auto ids = tokenizer.encode("hi", false);
    CELEG_TEST_CHECK(ids.size() == 1 && ids[0] == 3);
    CELEG_TEST_CHECK(tokenizer.decode(ids, false) == "hi");
    const auto contraction = tokenizer.encode("'S", false);
    CELEG_TEST_CHECK(contraction.size() == 1 && contraction[0] == 8);
    std::filesystem::remove(path);

    const auto gemma_path = std::filesystem::temp_directory_path() /
        "celeg_gemma_tokenizer_test.json";
    std::ofstream gemma(gemma_path);
    gemma << R"({
      "model": {
        "type": "BPE",
        "byte_fallback": true,
        "vocab": {
          "<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,
          "<|turn>":105,"<turn|>":106,
          "user":2364,"model":4368,"Hello":9259,
          "H":999,"h":1000,"e":1001,"l":1002,"o":1003,
          "u":1004,"s":1005,"r":1006,"m":1007,"d":1008,
          "<0x0A>":248,"▁":236743,"▁hello":29104
        },
        "merges": ["▁ h", "▁h e", "▁he l", "▁hel l", "▁hell o",
                   "H e", "He l", "Hel l", "Hell o",
                   "u s", "us e", "use r", "m o", "mo d", "mod e", "mode l"]
      },
      "normalizer": {"type":"Replace", "pattern":{"String":" "}, "content":"▁"},
      "added_tokens": [
        {"id":0,"content":"<pad>","special":true},
        {"id":1,"content":"<eos>","special":true},
        {"id":2,"content":"<bos>","special":true},
        {"id":105,"content":"<|turn>","special":true},
        {"id":106,"content":"<turn|>","special":true}
      ]
    })";
    gemma.close();

    celeg::BpeTokenizer gemma_tokenizer(
        gemma_path.string(),
        celeg::make_chat_template(celeg::ChatTemplateKind::Gemma4Instruct));
    CELEG_TEST_CHECK(gemma_tokenizer.bos_id() == 2);
    CELEG_TEST_CHECK(gemma_tokenizer.eos_id() == 1);
    CELEG_TEST_CHECK(gemma_tokenizer.pad_id() == 0);
    const auto leading_space = gemma_tokenizer.encode(" hello", false);
    CELEG_TEST_CHECK(leading_space.size() == 1 && leading_space[0] == 29104);
    const auto newline = gemma_tokenizer.encode("Hello\nHello", false);
    CELEG_TEST_CHECK(newline.size() == 3 && newline[0] == 9259 &&
                     newline[1] == 248 && newline[2] == 9259);

    const std::string gemma_chat = gemma_tokenizer.format_chat(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Hello"}}, true);
    CELEG_TEST_CHECK(gemma_chat ==
        "<bos><|turn>user\nHello<turn|>\n<|turn>model\n");
    const auto gemma_chat_ids = gemma_tokenizer.encode(gemma_chat, false);
    CELEG_TEST_CHECK(gemma_chat_ids.size() == 10);
    CELEG_TEST_CHECK(gemma_chat_ids[0] == 2 && gemma_chat_ids[1] == 105 &&
                     gemma_chat_ids[2] == 2364 && gemma_chat_ids[3] == 248 &&
                     gemma_chat_ids[4] == 9259 && gemma_chat_ids[5] == 106 &&
                     gemma_chat_ids[6] == 248 && gemma_chat_ids[7] == 105 &&
                     gemma_chat_ids[8] == 4368 && gemma_chat_ids[9] == 248);

    bool unsupported_rejected = false;
    try {
        (void)gemma_tokenizer.format_chat(
            std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "<|image>"}}, true);
    } catch (const std::invalid_argument&) {
        unsupported_rejected = true;
    }
    CELEG_TEST_CHECK(unsupported_rejected);
    unsupported_rejected = false;
    try {
        (void)gemma_tokenizer.format_chat(
            std::vector<celeg::ChatMessage>{{celeg::ChatRole::Developer, "no"}}, true);
    } catch (const std::invalid_argument&) {
        unsupported_rejected = true;
    }
    CELEG_TEST_CHECK(unsupported_rejected);
    unsupported_rejected = false;
    try {
        (void)gemma_tokenizer.encode("<|audio>", false);
    } catch (const std::invalid_argument&) {
        unsupported_rejected = true;
    }
    CELEG_TEST_CHECK(unsupported_rejected);
    std::filesystem::remove(gemma_path);
    std::cout << "tokenizer_test: ok\n";
}
