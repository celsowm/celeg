#pragma once

#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "celeg/runtime/checked_math.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace celeg {

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
               const ExecutionTopology& shape,
               const CompiledModelProgram& program)
        : page_tokens(page_tokens_value),
          attention_layers(shape.attention_layer_count) {
        if (page_tokens <= 0) {
            throw std::invalid_argument("PageLayout page_tokens must be positive");
        }
        layers.reserve(static_cast<size_t>(attention_layers));
        size_t vector_offset = 0;
        size_t scale_offset = 0;
        for (int slot = 0; slot < attention_layers; ++slot) {
            const int model_layer = slot < static_cast<int>(shape.layer_for_attention_slot.size())
                ? shape.layer_for_attention_slot[static_cast<size_t>(slot)] : slot;
            const AttentionSpec* attention = nullptr;
            if (model_layer >= 0 && model_layer < static_cast<int>(program.layers.size())) {
                const auto& mixer =
                    program.layers.at(static_cast<size_t>(model_layer)).mixer;
                if (const auto* compiled =
                        std::get_if<CompiledAttentionProgram>(&mixer)) {
                    attention = &compiled->semantics;
                }
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
                layers.push_back({layer_width, 1, true, latent->latent_rank,
                                  rotary_width, vector_offset, scale_offset});
                vector_offset += checked_multiply(
                    static_cast<size_t>(page_tokens),
                    static_cast<size_t>(layer_width),
                    "PageLayout latent vector overflow");
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
            vector_offset += checked_multiply(
                static_cast<size_t>(page_tokens),
                static_cast<size_t>(layer_kv_width),
                "PageLayout vector offset overflow");
            scale_offset += checked_multiply(
                static_cast<size_t>(page_tokens),
                static_cast<size_t>(layer_kv_heads),
                "PageLayout scale offset overflow");
        }
    }

    size_t page_vector_elements() const {
        size_t total = 0;
        for (const Layer& layer : layers) {
            const size_t layer_elements = checked_multiply(
                static_cast<size_t>(page_tokens), static_cast<size_t>(layer.kv_width),
                "PageLayout layer/token overflow");
            total = checked_add(total, layer_elements, "PageLayout vector overflow");
        }
        return total;
    }

    size_t page_scale_elements() const {
        size_t total = 0;
        for (const Layer& layer : layers) {
            const size_t layer_elements = checked_multiply(
                static_cast<size_t>(page_tokens), static_cast<size_t>(layer.kv_heads),
                "PageLayout scale layer/token overflow");
            total = checked_add(total, layer_elements, "PageLayout scale overflow");
        }
        return total;
    }

    size_t layer_vector_offset(int layer) const {
        return layers.at(static_cast<size_t>(layer)).vector_offset;
    }

    size_t layer_scale_offset(int layer) const {
        return layers.at(static_cast<size_t>(layer)).scale_offset;
    }

    size_t layer_vector_count(int layer, int used_tokens) const {
        return checked_multiply(
            static_cast<size_t>(used_tokens),
            static_cast<size_t>(layers.at(static_cast<size_t>(layer)).kv_width),
            "PageLayout layer vector count overflow");
    }

    size_t layer_scale_count(int layer, int used_tokens) const {
        return checked_multiply(
            static_cast<size_t>(used_tokens),
            static_cast<size_t>(layers.at(static_cast<size_t>(layer)).kv_heads),
            "PageLayout layer scale count overflow");
    }

    size_t page_vector_offset(uint32_t page) const {
        return checked_multiply(static_cast<size_t>(page), page_vector_elements(),
                                "PageLayout page vector offset overflow");
    }

    size_t page_scale_offset(uint32_t page) const {
        return checked_multiply(static_cast<size_t>(page), page_scale_elements(),
                                "PageLayout page scale offset overflow");
    }
};

}
