#pragma once

#include "celeg/model/resolved.hpp"

#include <cstddef>
#include <stdexcept>
#include <limits>

namespace celeg {

// Pure value type describing the per-page memory layout of a
// PhysicalPagedKvCache. Extracting the offset/stride math into a standalone
// struct lets future execution profiles supply a different layout (e.g. per-layer page
// tokens, or a different quantization group) at construction time without
// subclassing the cache itself (Open/Closed Principle). It also makes the
// layout testable in isolation from the CUDA resources (Single
// Responsibility Principle: the struct does math, the cache owns memory).
struct PageLayout {
    int page_tokens = 0;
    int attention_layers = 0;
    int kv_width = 0;
    int kv_heads = 0;

    PageLayout() = default;
    PageLayout(int page_tokens_value,
               const RuntimeTopology& shape)
        : page_tokens(page_tokens_value),
          attention_layers(shape.attention_layer_count),
          kv_width(shape.kv_width),
          kv_heads(shape.num_key_value_heads) {
        if (page_tokens <= 0) {
            throw std::invalid_argument("PageLayout page_tokens must be positive");
        }
        if (attention_layers <= 0) {
            throw std::invalid_argument("PageLayout requires at least one attention layer");
        }
    }

    // Number of bf16/int8 vector elements per page.
    size_t page_vector_elements() const {
        return checked_mul(
            checked_mul(static_cast<size_t>(attention_layers),
                        static_cast<size_t>(page_tokens),
                        "PageLayout layer/token overflow"),
            static_cast<size_t>(kv_width),
            "PageLayout vector overflow");
    }

    // Number of per-token scale elements per page (one per kv_head).
    size_t page_scale_elements() const {
        return checked_mul(
            checked_mul(static_cast<size_t>(attention_layers),
                        static_cast<size_t>(page_tokens),
                        "PageLayout scale layer/token overflow"),
            static_cast<size_t>(kv_heads),
            "PageLayout scale overflow");
    }

    // Byte offset of a layer's first token within a single page's vector
    // buffer. Used for per-layer COW copies and per-layer writes.
    size_t layer_vector_offset(int layer) const {
        return static_cast<size_t>(layer) *
               static_cast<size_t>(page_tokens) *
               static_cast<size_t>(kv_width);
    }

    size_t layer_scale_offset(int layer) const {
        return static_cast<size_t>(layer) *
               static_cast<size_t>(page_tokens) *
               static_cast<size_t>(kv_heads);
    }

    // Number of vector elements for `used_tokens` worth of K/V in one layer.
    size_t layer_vector_count(int used_tokens) const {
        return static_cast<size_t>(used_tokens) *
               static_cast<size_t>(kv_width);
    }

    size_t layer_scale_count(int used_tokens) const {
        return static_cast<size_t>(used_tokens) *
               static_cast<size_t>(kv_heads);
    }

    // Flat vector-element offset of page `page` (i.e. the first element of
    // layer 0, token 0 of that page).
    size_t page_vector_offset(uint32_t page) const {
        return static_cast<size_t>(page) * page_vector_elements();
    }

    size_t page_scale_offset(uint32_t page) const {
        return static_cast<size_t>(page) * page_scale_elements();
    }

private:
    static size_t checked_mul(size_t a, size_t b, const char* what) {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
            throw std::overflow_error(what);
        }
        return a * b;
    }
};

} // namespace celeg
