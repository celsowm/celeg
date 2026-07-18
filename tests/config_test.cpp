#include "lfm/config.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "lfm25-config-test.json";
    {
        std::ofstream out(path);
        out << R"({
          "model_type":"lfm2", "dtype":"bfloat16",
          "hidden_size":1024, "intermediate_size":2560,
          "num_hidden_layers":14, "num_attention_heads":16,
          "num_key_value_heads":8, "vocab_size":65536,
          "conv_L_cache":3, "conv_dim":1024,
          "max_position_embeddings":128000,
          "bos_token_id":1, "eos_token_id":7, "pad_token_id":0,
          "norm_eps":1e-5, "conv_bias":false,
          "tie_word_embeddings":true, "use_pos_enc":true,
          "rope_parameters":{"rope_theta":1000000.0,"rope_type":"default"},
          "layer_types":["conv","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv"]
        })";
    }

    const lfm::ModelConfig config = lfm::ModelConfig::load(path.string());
    config.validate_compiled_backend();
    assert(config.head_dim == 64);
    assert(config.layer_types.size() == 14);
    assert(config.summary().find("attention_layers=6") != std::string::npos);

    // The compiled backend must reject mathematically significant constants
    // that differ from the checkpoint it was specialized for.
    {
        std::ofstream out(path);
        out << R"({
          "model_type":"lfm2", "dtype":"bfloat16",
          "hidden_size":1024, "intermediate_size":2560,
          "num_hidden_layers":14, "num_attention_heads":16,
          "num_key_value_heads":8, "vocab_size":65536,
          "conv_L_cache":3, "conv_dim":1024,
          "max_position_embeddings":128000,
          "bos_token_id":1, "eos_token_id":7, "pad_token_id":0,
          "norm_eps":1e-4, "conv_bias":false,
          "tie_word_embeddings":true, "use_pos_enc":true,
          "rope_parameters":{"rope_theta":1000000.0,"rope_type":"default"},
          "layer_types":["conv","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv"]
        })";
    }
    bool rejected = false;
    try {
        lfm::ModelConfig::load(path.string()).validate_compiled_backend();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    std::filesystem::remove(path);
    std::cout << "config_test: ok\n";
}
