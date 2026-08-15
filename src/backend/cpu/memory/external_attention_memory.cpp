#include "celeg/backend/cpu/paged_kv.hpp"

#include <stdexcept>

namespace celeg {

CpuExternalAttentionMemory::CpuExternalAttentionMemory(
    CpuKvCacheMode mode, size_t page_tokens, size_t key_width,
    size_t value_width)
    : pool_(mode, page_tokens, CpuStatePageLayout{key_width, value_width, 0, 0}) {}

void CpuExternalAttentionMemory::append(const float* key, const float* value) {
    if (!key || !value) {
        throw std::invalid_argument("external attention memory pointers are required");
    }
    if (token_count_ % pool_.page_tokens() == 0) pages_.push_back(pool_.allocate());
    pool_.write(pages_.back(), token_count_ % pool_.page_tokens(), key, value);
    ++token_count_;
}

void CpuExternalAttentionMemory::clear() noexcept {
    for (CpuKvPageId page : pages_) {
        try { pool_.release(page); } catch (...) {}
    }
    pages_.clear();
    token_count_ = 0;
}

}
