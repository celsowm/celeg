#include "celeg/text/tokenizer.hpp"
#include "celeg/models/gemma4/chat_template.hpp"
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
        {"id":0,"content":"<|startoftext|>","special":true},
        {"id":99,"content":"<|image_pad|>","special":true}
      ]
    })";
    out.close();

    celeg::BpeTokenizer tokenizer(path.string());
    const auto ids = tokenizer.encode("hi", false);
    CELEG_TEST_CHECK(ids.size() == 1 && ids[0] == 3);
    CELEG_TEST_CHECK(tokenizer.decode(ids, false) == "hi");
    CELEG_TEST_CHECK(tokenizer.token_id("<|image_pad|>") == 99);
    const auto contraction = tokenizer.encode("'S", false);
    CELEG_TEST_CHECK(contraction.size() == 1 && contraction[0] == 8);
    std::filesystem::remove(path);

    celeg::TokenizerData checkpoint_data;
    checkpoint_data.tokens = {"h", "i", "hi"};
    checkpoint_data.merges = {"h i"};
    checkpoint_data.bos_id = 11;
    checkpoint_data.eos_id = 12;
    checkpoint_data.pad_id = 13;
    celeg::BpeTokenizer checkpoint_tokenizer(checkpoint_data);
    const auto checkpoint_ids = checkpoint_tokenizer.encode("hi", false);
    CELEG_TEST_CHECK(checkpoint_ids.size() == 1 && checkpoint_ids[0] == 2);
    CELEG_TEST_CHECK(checkpoint_tokenizer.bos_id() == 11 &&
                     checkpoint_tokenizer.eos_id() == 12 &&
                     checkpoint_tokenizer.pad_id() == 13);

    const auto lfm_path = std::filesystem::temp_directory_path() /
        "celeg_lfm2_tokenizer_test.json";
    std::ofstream lfm(lfm_path);
    lfm << R"({
      "model": {
        "type": "BPE",
        "vocab": {
          "hello":10,"Ġworld":11,"Ġ123":12,
          "456":13,"789":14,"0":15
        },
        "merges": []
      },
      "pre_tokenizer": {"type":"Split", "pattern": {
        "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
      }}
    })";
    lfm.close();
    celeg::BpeTokenizer lfm_tokenizer(lfm_path.string());
    const auto lfm_ids = lfm_tokenizer.encode("hello world 1234567890", false);
    CELEG_TEST_CHECK(lfm_ids.size() == 6 && lfm_ids[0] == 10 &&
                     lfm_ids[1] == 11 && lfm_ids[2] == 12 &&
                     lfm_ids[3] == 13 && lfm_ids[4] == 14 &&
                     lfm_ids[5] == 15);
    std::filesystem::remove(lfm_path);

    const auto granite_path = std::filesystem::temp_directory_path() /
        "celeg_granite_tokenizer_test.json";
    std::ofstream granite(granite_path);
    granite << R"({
      "model": {
        "type": "BPE",
        "vocab": {"1":20,"2":21,"3":22,"123":23,"'":24,"S":25},
        "merges": ["1 2", "12 3"]
      },
      "pre_tokenizer": {"type":"Split", "pattern": {
        "Regex": "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
      }}
    })";
    granite.close();
    celeg::BpeTokenizer granite_tokenizer(granite_path.string());
    const auto granite_numbers = granite_tokenizer.encode("123", false);
    CELEG_TEST_CHECK(granite_numbers.size() == 1 && granite_numbers[0] == 23);
    const auto granite_contraction = granite_tokenizer.encode("'S", false);
    CELEG_TEST_CHECK(granite_contraction.size() == 2 &&
                     granite_contraction[0] == 24 && granite_contraction[1] == 25);
    std::filesystem::remove(granite_path);

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

    celeg::BpeTokenizer gemma_tokenizer(gemma_path.string());
    const celeg::Gemma4InstructChatTemplate gemma_template;
    CELEG_TEST_CHECK(gemma_tokenizer.bos_id() == 2);
    CELEG_TEST_CHECK(gemma_tokenizer.eos_id() == 1);
    CELEG_TEST_CHECK(gemma_tokenizer.pad_id() == 0);
    const auto leading_space = gemma_tokenizer.encode(" hello", false);
    CELEG_TEST_CHECK(leading_space.size() == 1 && leading_space[0] == 29104);
    const auto newline = gemma_tokenizer.encode("Hello\nHello", false);
    CELEG_TEST_CHECK(newline.size() == 3 && newline[0] == 9259 &&
                     newline[1] == 248 && newline[2] == 9259);

    const std::string gemma_chat = celeg::render_chat(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Hello"}},
        gemma_template, true);
    CELEG_TEST_CHECK(gemma_chat ==
        "<bos><|turn>user\nHello<turn|>\n<|turn>model\n");
    const auto gemma_chat_ids = gemma_tokenizer.encode(gemma_chat, false);
    CELEG_TEST_CHECK(gemma_chat_ids.size() == 10);
    CELEG_TEST_CHECK(gemma_chat_ids[0] == 2 && gemma_chat_ids[1] == 105 &&
                     gemma_chat_ids[2] == 2364 && gemma_chat_ids[3] == 248 &&
                     gemma_chat_ids[4] == 9259 && gemma_chat_ids[5] == 106 &&
                     gemma_chat_ids[6] == 248 && gemma_chat_ids[7] == 105 &&
                     gemma_chat_ids[8] == 4368 && gemma_chat_ids[9] == 248);

    const std::string gemma_developer = celeg::render_chat(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::Developer, "no"}},
        gemma_template, true);
    CELEG_TEST_CHECK(gemma_developer.find("<|turn>system\nno") != std::string::npos);
    std::filesystem::remove(gemma_path);
    std::cout << "tokenizer_test: ok\n";
}
