#include "celeg/text/tokenizer.hpp"
#include "detail.hpp"

#include <cstdio>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace celeg {
using namespace tokenizer_detail;

std::vector<int32_t> BpeTokenizer::spm_score_tokenize(std::string_view normalized) const {
    struct Node {
        std::string text;
        int prev = -1;
        int next = -1;
        bool alive = true;
    };
    std::vector<Node> nodes;
    for (size_t offset = 0; offset < normalized.size();) {
        const auto [_, len] = next_cp(normalized, offset);
        const int index = static_cast<int>(nodes.size());
        nodes.push_back(Node{
            std::string(normalized.substr(offset, len)),
            index - 1,
            -1,
            true});
        if (index > 0) nodes[static_cast<size_t>(index - 1)].next = index;
        offset += len;
    }
    if (nodes.empty()) return {};

    struct Bigram {
        float score;
        int left;
        int right;
        size_t size;
    };
    struct BigramLess {
        bool operator()(const Bigram& a, const Bigram& b) const {
            return a.score < b.score || (a.score == b.score && a.left > b.left);
        }
    };
    std::priority_queue<Bigram, std::vector<Bigram>, BigramLess> queue;
    const auto try_add = [&](int left, int right) {
        if (left < 0 || right < 0 || !nodes[static_cast<size_t>(left)].alive ||
            !nodes[static_cast<size_t>(right)].alive ||
            nodes[static_cast<size_t>(left)].next != right) {
            return;
        }
        const std::string text =
            nodes[static_cast<size_t>(left)].text + nodes[static_cast<size_t>(right)].text;
        const auto it = vocab_.find(text);
        if (it == vocab_.end()) return;
        const float score = static_cast<size_t>(it->second) < id_score_.size()
            ? id_score_[static_cast<size_t>(it->second)] : 0.0f;
        queue.push(Bigram{score, left, right, text.size()});
    };
    for (size_t i = 1; i < nodes.size(); ++i) {
        try_add(static_cast<int>(i) - 1, static_cast<int>(i));
    }
    while (!queue.empty()) {
        const Bigram bigram = queue.top();
        queue.pop();
        Node& left = nodes[static_cast<size_t>(bigram.left)];
        Node& right = nodes[static_cast<size_t>(bigram.right)];
        if (!left.alive || !right.alive || left.text.size() + right.text.size() != bigram.size) {
            continue;
        }
        const int previous = left.prev;
        const int following = right.next;
        left.text += right.text;
        left.next = following;
        right.alive = false;
        if (following >= 0) nodes[static_cast<size_t>(following)].prev = bigram.left;
        try_add(previous, bigram.left);
        try_add(bigram.left, following);
    }

    std::vector<int32_t> ids;
    for (int index = 0; index >= 0;) {
        const Node& node = nodes[static_cast<size_t>(index)];
        const auto it = vocab_.find(node.text);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else if (byte_fallback_) {
            for (unsigned char byte : node.text) {
                char fallback[7] = {};
                std::snprintf(fallback, sizeof(fallback), "<0x%02X>", byte);
                const auto fallback_it = vocab_.find(fallback);
                if (fallback_it == vocab_.end()) {
                    throw std::runtime_error(
                        "SentencePiece byte-fallback token absent from vocabulary: " +
                        std::string(fallback));
                }
                ids.push_back(fallback_it->second);
            }
        } else {
            throw std::runtime_error(
                "SentencePiece produced a symbol absent from vocabulary: " + node.text);
        }
        index = node.next;
    }
    return ids;
}

}
