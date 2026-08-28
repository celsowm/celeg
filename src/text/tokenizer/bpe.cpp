#include "celeg/text/tokenizer.hpp"
#include "detail.hpp"

#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace celeg {
using namespace tokenizer_detail;

std::vector<std::string> BpeTokenizer::bpe(std::string_view encoded_piece) const {
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < encoded_piece.size()) {
        const auto [_, len] = next_cp(encoded_piece, i);
        symbols.emplace_back(encoded_piece.substr(i, len));
        i += len;
    }
    return bpe_symbols(std::move(symbols));
}

std::vector<std::string> BpeTokenizer::bpe_symbols(std::vector<std::string> symbols) const {
    if (symbols.size() < 2) return symbols;

    struct Node {
        std::string text;
        int prev = -1;
        int next = -1;
        bool alive = true;
    };
    struct Candidate {
        int32_t rank;
        int left;
        int right;
    };
    struct CandidateGreater {
        bool operator()(const Candidate& a, const Candidate& b) const {
            return a.rank > b.rank ||
                   (a.rank == b.rank && a.left > b.left);
        }
    };

    std::vector<Node> nodes;
    nodes.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size(); ++i) {
        nodes.push_back(Node{std::move(symbols[i]), static_cast<int>(i) - 1,
                             i + 1 < symbols.size() ? static_cast<int>(i) + 1 : -1,
                             true});
    }

    std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater> queue;
    const auto add_candidate = [&](int left, int right,
                                   auto& candidate_queue) {
        if (left < 0 || right < 0 || !nodes[left].alive || !nodes[right].alive ||
            nodes[left].next != right) return;
        const auto it = merge_rank_.find(pair_key(nodes[left].text, nodes[right].text));
        if (it != merge_rank_.end()) candidate_queue.push(Candidate{it->second, left, right});
    };
    for (size_t i = 1; i < nodes.size(); ++i) {
        add_candidate(static_cast<int>(i) - 1, static_cast<int>(i), queue);
    }

    while (!queue.empty()) {
        const Candidate candidate = queue.top();
        queue.pop();
        if (!nodes[candidate.left].alive || !nodes[candidate.right].alive ||
            nodes[candidate.left].next != candidate.right ||
            nodes[candidate.right].prev != candidate.left) {
            continue;
        }
        const auto current = merge_rank_.find(
            pair_key(nodes[candidate.left].text, nodes[candidate.right].text));
        if (current == merge_rank_.end() || current->second != candidate.rank) continue;

        Node& left = nodes[candidate.left];
        Node& right = nodes[candidate.right];
        const int previous = left.prev;
        const int following = right.next;
        left.text += right.text;
        left.next = following;
        right.alive = false;
        if (following >= 0) nodes[following].prev = candidate.left;
        add_candidate(previous, candidate.left, queue);
        add_candidate(candidate.left, following, queue);
    }

    std::vector<std::string> result;
    for (int index = 0; index >= 0;) {
        if (nodes[index].alive) result.push_back(std::move(nodes[index].text));
        index = nodes[index].next;
    }
    return result;
}

/// SentencePiece's own tokenizer (llama.cpp's "llama" GGUF vocab type) has no
/// explicit merge-rank table: any substring that is itself a vocabulary entry
/// is a valid merge, prioritized by that entry's own score (higher merges
/// first; ties broken toward the leftmost position). This mirrors

}
