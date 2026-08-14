#pragma once

#include "celeg/backend/cuda/packed/session.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace celeg {

// Tracks the typed session identity of metadata bindings currently staged in
// the packed workspace. Storage generation is part of the key because a
// session can retain its identity while replacing its backing cache allocation.
class PackedMetadataCache {
public:
    explicit PackedMetadataCache(size_t capacity) { keys_.reserve(capacity); }

    bool changed(std::span<const PackedSessionContext> sessions) const {
        if (keys_.size() != sessions.size()) return true;
        for (size_t index = 0; index < sessions.size(); ++index) {
            if (keys_[index].session != sessions[index].owner.session_identity ||
                keys_[index].storage_generation !=
                    sessions[index].storage_generation_value) {
                return true;
            }
        }
        return false;
    }

    void update(std::span<const PackedSessionContext> sessions) {
        keys_.resize(sessions.size());
        for (size_t index = 0; index < sessions.size(); ++index) {
            keys_[index] = {
                sessions[index].owner.session_identity,
                sessions[index].storage_generation_value,
            };
        }
    }

private:
    struct Key {
        const SessionState* session = nullptr;
        uint64_t storage_generation = 0;
    };
    std::vector<Key> keys_;
};

} // namespace celeg
