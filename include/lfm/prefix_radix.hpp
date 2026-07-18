#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lfm {

class PrefixRadixIndex {
public:
    using EntryId = uint64_t;

    struct Match {
        EntryId id = 0;
        size_t matched_tokens = 0;
    };

    PrefixRadixIndex();
    ~PrefixRadixIndex();

    PrefixRadixIndex(const PrefixRadixIndex&) = delete;
    PrefixRadixIndex& operator=(const PrefixRadixIndex&) = delete;

    void insert(const std::vector<int32_t>& tokens, EntryId id);
    bool erase(const std::vector<int32_t>& tokens, EntryId id);
    std::optional<Match> longest_prefix(const std::vector<int32_t>& tokens) const;
    void clear();
    size_t size() const { return size_; }
    size_t node_count() const { return nodes_; }

private:
    struct Node {
        std::unordered_map<int32_t, std::unique_ptr<Node>> children;
        std::vector<EntryId> terminal_ids;
    };

    static bool erase_recursive(Node& node,
                                const std::vector<int32_t>& tokens,
                                size_t depth,
                                EntryId id,
                                bool* erased,
                                size_t* removed_nodes);

    std::unique_ptr<Node> root_;
    size_t size_ = 0;
    size_t nodes_ = 1;
};

} // namespace lfm
