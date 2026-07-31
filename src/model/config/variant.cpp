#include "celeg/model/config/variant.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace celeg {

namespace {

bool close_float(float a, float b, float tol) {
    const float diff = a > b ? a - b : b - a;
    return diff <= tol;
}

bool repo_hint_contains(std::string_view repo_hint, std::string_view needle) {
    if (repo_hint.empty() || needle.empty()) return false;
    std::string haystack;
    haystack.reserve(repo_hint.size());
    for (char c : repo_hint) haystack.push_back(static_cast<char>(::tolower(c)));
    std::string sub;
    sub.reserve(needle.size());
    for (char c : needle) sub.push_back(static_cast<char>(::tolower(c)));
    return haystack.find(sub) != std::string::npos;
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

bool Lfm25_1_2B_Instruct_Variant::matches(const ModelShape& shape,
                                           std::string_view repo_hint) const {
    // Disambiguate from the Thinking variant which shares this exact topology.
    // When a repo hint is present, reject the Thinking checkpoint; otherwise
    // fall back to the shape-only match so shape-only loading keeps selecting
    // the Instruct variant.
    if (!repo_hint.empty() && repo_hint_contains(repo_hint, "thinking")) {
        return false;
    }
    return matches(shape);
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
// LFM2.5-1.2B-Thinking
// ---------------------------------------------------------------------------

std::string_view Lfm25_1_2B_Thinking_Variant::id() const {
    return "lfm2.5-1.2b-thinking";
}
std::string_view Lfm25_1_2B_Thinking_Variant::repo_id() const {
    return "LiquidAI/LFM2.5-1.2B-Thinking";
}

bool Lfm25_1_2B_Thinking_Variant::matches(const ModelShape& shape) const {
    // Never match on shape alone: the Thinking and Instruct checkpoints share
    // an identical topology, so a shape-only match would make the registry
    // ambiguous. Selection must go through the (shape, repo_hint) overload.
    (void)shape;
    return false;
}

bool Lfm25_1_2B_Thinking_Variant::matches(const ModelShape& shape,
                                           std::string_view repo_hint) const {
    // Same topology as the Instruct variant; require the repo hint to confirm
    // this is the Thinking checkpoint.
    if (!repo_hint_contains(repo_hint, "thinking")) return false;
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

ModelShape Lfm25_1_2B_Thinking_Variant::resolve_shape(ModelShape shape) const {
    // Same intermediate_size=8192 correction as the Instruct variant.
    shape.intermediate = 8192;
    return shape;
}

ChatTemplateKind Lfm25_1_2B_Thinking_Variant::chat_template_kind() const {
    return ChatTemplateKind::Lfm2Instruct;
}

std::string Lfm25_1_2B_Thinking_Variant::label() const {
    return "LiquidAI LFM2.5-1.2B-Thinking";
}

// ---------------------------------------------------------------------------
// LFM2.5-8B-A1B (LFM2 MoE architecture)
// ---------------------------------------------------------------------------

std::string_view Lfm25_8B_A1B_Variant::id() const {
    return "lfm2.5-8b-a1b";
}
std::string_view Lfm25_8B_A1B_Variant::repo_id() const {
    return "LiquidAI/LFM2.5-8B-A1B";
}

bool Lfm25_8B_A1B_Variant::matches(const ModelShape& shape) const {
    // The MoE topology is unique: no dense variant shares these dimensions or
    // the MoE discriminator, so a shape-only match is unambiguous.
    return shape.architecture == ArchitectureKind::MoeLfm2 &&
           shape.hidden == 2048 &&
           shape.intermediate == 7168 &&
           shape.num_hidden_layers == 24 &&
           shape.num_attention_heads == 32 &&
           shape.num_key_value_heads == 8 &&
           shape.head_dim == 64 &&
           shape.vocab_size == 128000 &&
           shape.conv_cache == 3 &&
           shape.conv_dim == 2048 &&
           shape.attention_layer_count == 6 &&
           shape.conv_layer_count == 18 &&
           shape.num_dense_layers == 2 &&
           shape.num_experts == 32 &&
           shape.experts_per_token == 4 &&
           shape.moe_intermediate == 1792 &&
           !shape.use_expert_bias &&
           shape.normalize_topk &&
           close_float(shape.routed_scaling_factor, 1.0f, 1.0e-6f) &&
           close_float(shape.norm_eps, 1.0e-5f, 1.0e-12f) &&
           close_float(shape.rope_theta, 5'000'000.0f, 0.5f);
}

bool Lfm25_8B_A1B_Variant::matches(const ModelShape& shape,
                                   std::string_view repo_hint) const {
    // Prefer an exact repo-hint match when available, but fall back to the
    // shape-only match so shape-only loading still works.
    if (!repo_hint.empty()) {
        const bool hint_ok = repo_hint_contains(repo_hint, "8b-a1b") ||
                             repo_hint_contains(repo_hint, "8b_a1b") ||
                             repo_hint_contains(repo_hint, "a1b");
        if (hint_ok && !matches(shape)) return false;
        if (hint_ok) return true;
    }
    return matches(shape);
}

ModelShape Lfm25_8B_A1B_Variant::resolve_shape(ModelShape shape) const {
    // shape.intermediate already holds the dense FFN intermediate size
    // (config intermediate_size = 7168). Pin it explicitly so weight loading
    // and buffer allocation use the dense value for the first num_dense_layers.
    shape.intermediate = shape.dense_intermediate;
    return shape;
}

ChatTemplateKind Lfm25_8B_A1B_Variant::chat_template_kind() const {
    return ChatTemplateKind::Lfm2Instruct;
}

std::string Lfm25_8B_A1B_Variant::label() const {
    return "LiquidAI LFM2.5-8B-A1B (MoE)";
}

// ---------------------------------------------------------------------------
// LFM2-8B-A1B (earlier LFM2 MoE release, vocab 65536 / rope_theta 1e6)
// ---------------------------------------------------------------------------

std::string_view Lfm25_8B_A1B_LFM2_Variant::id() const {
    return "lfm2-8b-a1b";
}

std::string_view Lfm25_8B_A1B_LFM2_Variant::repo_id() const {
    return "LiquidAI/LFM2-8B-A1B";
}

bool Lfm25_8B_A1B_LFM2_Variant::matches(const ModelShape& shape) const {
    return shape.architecture == ArchitectureKind::MoeLfm2 &&
           shape.hidden == 2048 &&
           shape.intermediate == 7168 &&
           shape.num_hidden_layers == 24 &&
           shape.num_attention_heads == 32 &&
           shape.num_key_value_heads == 8 &&
           shape.head_dim == 64 &&
           shape.vocab_size == 65536 &&
           shape.conv_cache == 3 &&
           shape.conv_dim == 2048 &&
           shape.attention_layer_count == 6 &&
           shape.conv_layer_count == 18 &&
           shape.num_dense_layers == 2 &&
           shape.num_experts == 32 &&
           shape.experts_per_token == 4 &&
           shape.moe_intermediate == 1792 &&
           shape.use_expert_bias &&
           shape.normalize_topk &&
           close_float(shape.routed_scaling_factor, 1.0f, 1.0e-6f) &&
           close_float(shape.norm_eps, 1.0e-5f, 1.0e-12f) &&
           close_float(shape.rope_theta, 1'000'000.0f, 0.5f);
}

bool Lfm25_8B_A1B_LFM2_Variant::matches(const ModelShape& shape,
                                        std::string_view repo_hint) const {
    if (!repo_hint.empty()) {
        const bool hint_ok = repo_hint_contains(repo_hint, "8b-a1b") ||
                             repo_hint_contains(repo_hint, "8b_a1b") ||
                             repo_hint_contains(repo_hint, "a1b");
        if (hint_ok && !matches(shape)) return false;
        if (hint_ok) return true;
    }
    return matches(shape);
}

ModelShape Lfm25_8B_A1B_LFM2_Variant::resolve_shape(ModelShape shape) const {
    shape.intermediate = shape.dense_intermediate;
    return shape;
}

ChatTemplateKind Lfm25_8B_A1B_LFM2_Variant::chat_template_kind() const {
    return ChatTemplateKind::Lfm2Instruct;
}

std::string Lfm25_8B_A1B_LFM2_Variant::label() const {
    return "LiquidAI LFM2-8B-A1B (MoE)";
}

std::string_view Granite_Variant::id() const { return "granite"; }
std::string_view Granite_Variant::repo_id() const {
    return "ibm-granite/granite-4.1-8b";
}
bool Granite_Variant::matches(const ModelShape& shape) const {
    return shape.architecture == ArchitectureKind::Granite;
}
ChatTemplateKind Granite_Variant::chat_template_kind() const {
    return ChatTemplateKind::Lfm2Instruct;
}
std::string Granite_Variant::label() const { return "IBM Granite 4.1"; }

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
        registry.register_variant(std::make_unique<Lfm25_1_2B_Thinking_Variant>());
        registry.register_variant(std::make_unique<Lfm25_8B_A1B_Variant>());
        registry.register_variant(std::make_unique<Lfm25_8B_A1B_LFM2_Variant>());
        registry.register_variant(std::make_unique<Granite_Variant>());
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
    return select(shape, {});
}

const IModelVariant& ModelVariantRegistry::select(const ModelShape& shape,
                                                  std::string_view repo_hint) const {
    const IModelVariant* match = nullptr;
    for (const auto& variant : variants_) {
        if (variant->matches(shape, repo_hint)) {
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

} // namespace celeg
