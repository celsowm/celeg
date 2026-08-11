#pragma once

#include "celeg/model/resolved.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace celeg {

// Pure value type describing the per-page memory layout of a
// PhysicalPagedKvCache. Extracting the offset/stride math into a standalone
// struct lets future execution profiles supply a different layout (e.g. per-layer page
// tokens, or a different quantization group) at construction time without
// subclassing the cache itself. It also makes the
// layout testable in isolation from the CUDA resources (Single
// Responsibility Principle: the struct does math, the cache owns memory).
struct PageLayout {
    struct Layer {
        int kv_width = 0;
        int kv_heads = 0;
        bool latent = false;
        int latent_rank = 0;
        int rotary_width = 0;
        size_t vector_offset = 0;
        size_t scale_offset = 0;
    };

    int page_tokens = 0;
    int attention_layers = 0;
    std::vector<Layer> layers;

    PageLayout() = default;
    PageLayout(int page_tokens_value,
               const RuntimeTopology& shape)
        : page_tokens(page_tokens_value),
          attention_layers(shape.attention_layer_count) {
        if (page_tokens <= 0) {
            throw std::invalid_argument("PageLayout page_tokens must be positive");
        }
        // A pure recurrent model has no KV layers. Keep an empty layout so
        // the shared page allocator can still provide request lifetime and
        // prefix-cache bookkeeping without allocating unused KV storage.
        layers.reserve(static_cast<size_t>(attention_layers));
        size_t vector_offset = 0;
        size_t scale_offset = 0;
        for (int slot = 0; slot < attention_layers; ++slot) {
            const int model_layer = slot < static_cast<int>(shape.layer_for_attention_slot.size())
                ? shape.layer_for_attention_slot[static_cast<size_t>(slot)] : slot;
            const AttentionSpec* attention = nullptr;
            if (model_layer >= 0 && model_layer < static_cast<int>(shape.attention_layouts.size())) {
                attention = &shape.attention_layout(model_layer);
            }
            if (!attention) {
                throw std::invalid_argument("PageLayout requires an attention layout for every slot");
            }
            if (const auto* latent = attention->latent_state()) {
                const int rotary_width = latent->decoupled_rope ? latent->rope_head_dim : 0;
                const int layer_width = latent->latent_rank + rotary_width;
                if (latent->latent_rank <= 0 || layer_width <= 0) {
                    throw std::invalid_argument("PageLayout latent layer has invalid dimensions");
                }
                // The key pool stores [latent key | rotary key], while the
                // value pool uses the first latent_rank elements. Keeping a
                // common stride preserves page-table and COW invariants.
                layers.push_back({layer_width, 1, true, latent->latent_rank,
                                  rotary_width, vector_offset, scale_offset});
                vector_offset += static_cast<size_t>(page_tokens) *
                    static_cast<size_t>(layer_width);
                scale_offset += static_cast<size_t>(page_tokens);
                continue;
            }
            const int layer_kv_width = attention->key_value_width();
            const int layer_kv_heads = attention->key_value_heads;
            if (layer_kv_width <= 0 || layer_kv_heads <= 0) {
                throw std::invalid_argument("PageLayout layer has invalid KV dimensions");
            }
            layers.push_back({layer_kv_width, layer_kv_heads, false, 0, 0,
                              vector_offset, scale_offset});
            vector_offset += static_cast<size_t>(page_tokens) *
                static_cast<size_t>(layer_kv_width);
            scale_offset += static_cast<size_t>(page_tokens) *
                static_cast<size_t>(layer_kv_heads);
        }
    }

    // Number of bf16/int8 vector elements per page.
    size_t page_vector_elements() const {
        size_t total = 0;
        for (const Layer& layer : layers) {
            const size_t layer_elements = checked_mul(
                static_cast<size_t>(page_tokens), static_cast<size_t>(layer.kv_width),
                "PageLayout layer/token overflow");
            if (layer_elements > std::numeric_limits<size_t>::max() - total) {
                throw std::overflow_error("PageLayout vector overflow");
            }
            total += layer_elements;
        }
        return total;
    }

    // Number of per-token scale elements per page (one per kv_head).
    size_t page_scale_elements() const {
        size_t total = 0;
        for (const Layer& layer : layers) {
            const size_t layer_elements = checked_mul(
                static_cast<size_t>(page_tokens), static_cast<size_t>(layer.kv_heads),
                "PageLayout scale layer/token overflow");
            if (layer_elements > std::numeric_limits<size_t>::max() - total) {
                throw std::overflow_error("PageLayout scale overflow");
            }
            total += layer_elements;
        }
        return total;
    }

    // Byte offset of a layer's first token within a single page's vector
    // buffer. Used for per-layer COW copies and per-layer writes.
    size_t layer_vector_offset(int layer) const {
        return layers.at(static_cast<size_t>(layer)).vector_offset;
    }

    size_t layer_scale_offset(int layer) const {
        return layers.at(static_cast<size_t>(layer)).scale_offset;
    }

    size_t layer_vector_count(int layer, int used_tokens) const {
        return static_cast<size_t>(used_tokens) *
               static_cast<size_t>(layers.at(static_cast<size_t>(layer)).kv_width);
    }

    size_t layer_scale_count(int layer, int used_tokens) const {
        return static_cast<size_t>(used_tokens) *
               static_cast<size_t>(layers.at(static_cast<size_t>(layer)).kv_heads);
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
