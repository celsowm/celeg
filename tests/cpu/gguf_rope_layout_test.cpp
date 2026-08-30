#include "celeg/checkpoint/gguf_position_profile.hpp"
#include "celeg/model/definition.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace {

/// llama.cpp's convert_hf_to_gguf.py permutes an attention row block by
/// reshaping it to (2, head_dim/2), swapping those axes and flattening again.
/// An original row `a * half + b` therefore lands at `2 * b + a`, so the row
/// stored at `destination` came from `(destination % 2) * half + destination / 2`.
/// celeg never performs this reordering; it expresses the same fact as a
/// RopePairingKind so both backends rotate the components actually stored.
int llama_cpp_permuted_source_row(int destination, int head_dim) {
    const int half = head_dim / 2;
    return (destination % 2) * half + destination / 2;
}

/// Rotating the permuted rows with AdjacentPairs must produce exactly the
/// permutation of rotating the original rows with SplitHalf. This is the
/// invariant that makes "reorder rows at load + SplitHalf" and "leave rows
/// alone + AdjacentPairs" interchangeable, and therefore the justification for
/// deleting the load-time reordering entirely.
void adjacent_pairing_matches_the_permuted_row_order() {
    constexpr int head_dim = 8;
    constexpr int half = head_dim / 2;
    const std::vector<float> original = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    const float cosines[half] = {0.9f, 0.8f, 0.7f, 0.6f};
    const float sines[half] = {0.1f, 0.2f, 0.3f, 0.4f};

    /// Reference: rotate the HuggingFace order with SplitHalf pairing.
    std::vector<float> split_rotated = original;
    for (int pair = 0; pair < half; ++pair) {
        const float a = original[pair];
        const float b = original[half + pair];
        split_rotated[pair] = a * cosines[pair] - b * sines[pair];
        split_rotated[half + pair] = b * cosines[pair] + a * sines[pair];
    }

    /// What celeg now does: rotate llama.cpp's stored order with AdjacentPairs.
    std::vector<float> permuted(head_dim);
    for (int row = 0; row < head_dim; ++row) {
        permuted[row] = original[llama_cpp_permuted_source_row(row, head_dim)];
    }
    std::vector<float> adjacent_rotated = permuted;
    for (int pair = 0; pair < half; ++pair) {
        const float a = permuted[2 * pair];
        const float b = permuted[2 * pair + 1];
        adjacent_rotated[2 * pair] = a * cosines[pair] - b * sines[pair];
        adjacent_rotated[2 * pair + 1] = b * cosines[pair] + a * sines[pair];
    }

    for (int row = 0; row < head_dim; ++row) {
        const float expected =
            split_rotated[llama_cpp_permuted_source_row(row, head_dim)];
        CELEG_TEST_CHECK(std::abs(adjacent_rotated[row] - expected) < 1e-5f);
    }
}

/// The architecture list mirrors llama.cpp's llama_model_rope_type(). These are
/// the cases the GGUF corpus in docs/inference_report.md actually exercises, and
/// the ones a regression would silently corrupt: getting any of them wrong
/// mismatches Q/K against the rotation and destroys attention.
void architecture_table_matches_llama_cpp() {
    const std::vector<std::string> adjacent = {
        "llama", "minicpm", "nanbeige", "granite", "granitemoe",
        "deepseek2", "internlm2", "command-r", "starcoder", "olmo",
    };
    for (const std::string& architecture : adjacent) {
        CELEG_TEST_CHECK(
            celeg::gguf_architecture_uses_adjacent_rope_pairs(architecture));
    }

    /// Everything else -- notably the NEOX-style architectures -- keeps the
    /// original HuggingFace row order. This half of the table is what the
    /// previous blanket "every GGUF attn_q/attn_k is permuted" heuristic got
    /// wrong, so it matters as much as the list above.
    const std::vector<std::string> split_half = {
        "lfm2", "lfm2moe", "qwen2", "qwen3", "falcon", "gptneox",
        "phi3", "gemma", "gemma2", "stablelm", "bitnet", "unknown-architecture",
    };
    for (const std::string& architecture : split_half) {
        CELEG_TEST_CHECK(
            !celeg::gguf_architecture_uses_adjacent_rope_pairs(architecture));
    }
}

/// An architecture that never applies RoPE cannot meaningfully claim a pairing;
/// the two GGUF format predicates must not both be true for one architecture.
void rope_free_architectures_are_not_also_adjacent() {
    const std::vector<std::string> rope_free = {
        "mamba", "mamba2", "jamba", "rwkv6", "rwkv7", "gpt2", "bloom", "t5",
    };
    for (const std::string& architecture : rope_free) {
        CELEG_TEST_CHECK(celeg::gguf_architecture_never_applies_rope(architecture));
        CELEG_TEST_CHECK(
            !celeg::gguf_architecture_uses_adjacent_rope_pairs(architecture));
    }
}

}

int main() {
    adjacent_pairing_matches_the_permuted_row_order();
    architecture_table_matches_llama_cpp();
    rope_free_architectures_are_not_also_adjacent();
    return 0;
}
