#include "lfm/model_variant.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lfm {

namespace {

bool close_float(float a, float b, float tol) {
    const float diff = a > b ? a - b : b - a;
    return diff <= tol;
}

} // namespace

// ---------------------------------------------------------------------------
// Built-in variants
// ---------------------------------------------------------------------------

std::string_view Lfm25_230M_Variant::id() const { return "lfm2.5-230m"; }
std::string_view Lfm25_230M_Variant::repo_id() const { return "LiquidAI/LFM2.5-230M"; }

bool Lfm25_230M_Variant::matches(const ModelShape& shape) const {
    return shape.hidden == 1024 &&
           shape.intermediate == 2560 &&
           shape.num_hidden_layers == 14 &&
           shape.num_attention_heads == 16 &&
           shape.num_key_value_heads == 8 &&
           shape.head_dim == 64 &&
           shape.vocab_size == 65536 &&
           shape.conv_cache == 3 &&
           shape.conv_dim == 1024 &&
           shape.attention_layer_count == 6 &&
           shape.conv_layer_count == 8 &&
           close_float(shape.norm_eps, 1.0e-5f, 1.0e-12f) &&
           close_float(shape.rope_theta, 1'000'000.0f, 0.5f);
}

ChatTemplateKind Lfm25_230M_Variant::chat_template_kind() const {
    return ChatTemplateKind::Lfm2Instruct;
}

std::string Lfm25_230M_Variant::label() const {
    return "LiquidAI LFM2.5-230M";
}

std::string_view Lfm25_1_2B_Instruct_Variant::id() const {
    return "lfm2.5-1.2b-instruct";
}
std::string_view Lfm25_1_2B_Instruct_Variant::repo_id() const {
    return "LiquidAI/LFM2.5-1.2B-Instruct";
}

bool Lfm25_1_2B_Instruct_Variant::matches(const ModelShape& shape) const {
    // LiquidAI/LFM2.5-1.2B-Instruct has 16 layers: 6 full-attention and
    // 10 convolutional, with 32 query heads and 8 KV heads at head_dim 64.
    // config.json publishes intermediate_size=12288 but the published
    // checkpoint actually stores w1/w3 with 8192 rows (the SwiGLU
    // block_auto_adjust_ff_dim flag re-derives the real value). Accept
    // either shape so variant selection succeeds from raw config alone;
    // resolve_shape() then canonicalizes to the real 8192.
    return shape.hidden == 2048 &&
           (shape.intermediate == 12288 || shape.intermediate == 8192) &&
           shape.num_hidden_layers == 16 &&
           shape.num_attention_heads == 32 &&
           shape.num_key_value_heads == 8 &&
           shape.head_dim == 64 &&
           shape.vocab_size == 65536 &&
           shape.conv_cache == 3 &&
           shape.conv_dim == 2048 &&
           shape.attention_layer_count == 6 &&
           shape.conv_layer_count == 10 &&
           close_float(shape.norm_eps, 1.0e-5f, 1.0e-12f) &&
           close_float(shape.rope_theta, 1'000'000.0f, 0.5f);
}

ModelShape Lfm25_1_2B_Instruct_Variant::resolve_shape(ModelShape shape) const {
    // The published checkpoint stores w1/w3 with 8192 rows even though
    // config.json advertises intermediate_size=12288 (see matches() above).
    // Force the runtime topology to 8192 so weight loading and buffer
    // allocation use the real tensor dimensions.
    shape.intermediate = 8192;
    return shape;
}

ChatTemplateKind Lfm25_1_2B_Instruct_Variant::chat_template_kind() const {
    return ChatTemplateKind::Lfm2Instruct;
}

std::string Lfm25_1_2B_Instruct_Variant::label() const {
    return "LiquidAI LFM2.5-1.2B-Instruct";
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

ModelVariantRegistry& ModelVariantRegistry::instance() {
    static ModelVariantRegistry registry;
    // Lazy self-registration on first access. The registry's variants_ vector
    // is empty only before the first call, so this branch runs exactly once
    // per process. No std::call_once / std::mutex needed: Meyers singleton
    // construction is already thread-safe.
    if (registry.variants_.empty()) {
        registry.register_variant(std::make_unique<Lfm25_230M_Variant>());
        registry.register_variant(std::make_unique<Lfm25_1_2B_Instruct_Variant>());
    }
    return registry;
}

void ModelVariantRegistry::register_variant(std::unique_ptr<IModelVariant> variant) {
    if (!variant) throw std::invalid_argument("variant is null");
    const std::string_view id = variant->id();
    for (const auto& existing : variants_) {
        if (existing->id() == id) {
            throw std::invalid_argument("duplicate model variant id: " + std::string(id));
        }
    }
    variants_.push_back(std::move(variant));
}

const IModelVariant* ModelVariantRegistry::find(std::string_view id) const {
    for (const auto& variant : variants_) {
        if (variant->id() == id) return variant.get();
    }
    return nullptr;
}

const IModelVariant& ModelVariantRegistry::select(const ModelShape& shape) const {
    const IModelVariant* match = nullptr;
    for (const auto& variant : variants_) {
        if (variant->matches(shape)) {
            if (match != nullptr) {
                throw std::runtime_error(
                    "multiple model variants match the same shape; registry is ambiguous");
            }
            match = variant.get();
        }
    }
    if (match == nullptr) {
        throw std::runtime_error(
            "no registered model variant matches checkpoint shape: " + shape.summary());
    }
    return *match;
}

std::vector<std::string_view> ModelVariantRegistry::ids() const {
    std::vector<std::string_view> result;
    result.reserve(variants_.size());
    for (const auto& variant : variants_) {
        result.push_back(variant->id());
    }
    return result;
}

void register_builtin_variants() {
    // Touching instance() is enough to lazy-register the built-ins.
    (void)ModelVariantRegistry::instance();
}

} // namespace lfm
