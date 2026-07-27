#include "lfm/runtime/cache/prefix_radix.hpp"

#include <algorithm>
#include <stdexcept>

namespace lfm {

PrefixRadixIndex::PrefixRadixIndex() : root_(std::make_unique<Node>()) {}
PrefixRadixIndex::~PrefixRadixIndex() = default;

void PrefixRadixIndex::insert(const std::vector<int32_t>& tokens, EntryId id) {
    if (tokens.empty()) throw std::invalid_argument("prefix radix rejects empty prefixes");
    if (id == 0) throw std::invalid_argument("prefix radix entry ID must be non-zero");
    Node* node = root_.get();
    for (int32_t token : tokens) {
        auto [it, inserted] = node->children.try_emplace(token);
        if (inserted) {
            it->second = std::make_unique<Node>();
            ++nodes_;
        }
        node = it->second.get();
    }
    if (std::find(node->terminal_ids.begin(), node->terminal_ids.end(), id) !=
        node->terminal_ids.end()) {
        throw std::invalid_argument("prefix radix already contains entry ID at prefix");
    }
    node->terminal_ids.push_back(id);
    ++size_;
}

bool PrefixRadixIndex::erase_recursive(Node& node,
                                       const std::vector<int32_t>& tokens,
                                       size_t depth,
                                       EntryId id,
                                       bool* erased,
                                       size_t* removed_nodes) {
    if (depth == tokens.size()) {
        auto it = std::find(node.terminal_ids.begin(), node.terminal_ids.end(), id);
        if (it == node.terminal_ids.end()) return false;
        node.terminal_ids.erase(it);
        *erased = true;
    } else {
        auto child = node.children.find(tokens[depth]);
        if (child == node.children.end()) return false;
        const bool remove_child = erase_recursive(*child->second, tokens, depth + 1,
                                                  id, erased, removed_nodes);
        if (remove_child) {
            node.children.erase(child);
            ++(*removed_nodes);
        }
    }
    return depth != 0 && node.terminal_ids.empty() && node.children.empty();
}

bool PrefixRadixIndex::erase(const std::vector<int32_t>& tokens, EntryId id) {
    if (tokens.empty() || id == 0) return false;
    bool erased = false;
    size_t removed_nodes = 0;
    (void)erase_recursive(*root_, tokens, 0, id, &erased, &removed_nodes);
    if (erased) {
        --size_;
        nodes_ -= removed_nodes;
    }
    return erased;
}

std::optional<PrefixRadixIndex::Match>
PrefixRadixIndex::longest_prefix(const std::vector<int32_t>& tokens) const {
    const Node* node = root_.get();
    std::optional<Match> best;
    size_t depth = 0;
    for (int32_t token : tokens) {
        auto it = node->children.find(token);
        if (it == node->children.end()) break;
        node = it->second.get();
        ++depth;
        if (!node->terminal_ids.empty()) {
            // Multiple cache records may exist at one prefix. The engine keeps
            // entries unique, but selecting the newest ID is deterministic and
            // makes the index robust to transitional duplicates.
            best = Match{node->terminal_ids.back(), depth};
        }
    }
    return best;
}

void PrefixRadixIndex::clear() {
    root_ = std::make_unique<Node>();
    size_ = 0;
    nodes_ = 1;
}

} // namespace lfm
