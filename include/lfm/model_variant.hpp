#pragma once

#include "lfm/model_shape.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lfm {

// Chat template identifier selected by the variant. Today only the LFM2
// Instruct template is supported; future variants can introduce new ones.
enum class ChatTemplateKind : uint8_t {
    Lfm2Instruct,
};

// Open/Closed Principle: new variants are added by registering a new
// IModelVariant subclass rather than editing the runtime. The runtime
// depends only on this interface.
class IModelVariant {
public:
    virtual ~IModelVariant() = default;
    // Stable, lowercase identifier (e.g. "lfm2.5-230m").
    virtual std::string_view id() const = 0;
    // Canonical HuggingFace repository id.
    virtual std::string_view repo_id() const = 0;
    // True if the shape matches this variant. The registry picks the variant
    // whose matches() returns true; exactly one is expected to match.
    virtual bool matches(const ModelShape& shape) const = 0;
    // Two-argument overload used by the registry to disambiguate checkpoints
    // that share an identical topology. `repo_hint` is the source repo id
    // captured from config.json's optional "_name" field (may be empty).
    // Variants with unique shapes keep the single-argument semantics by
    // ignoring `repo_hint`. The default delegates to the single-argument form
    // so existing variants stay compatible.
    virtual bool matches(const ModelShape& shape, std::string_view repo_hint) const {
        (void)repo_hint;
        return matches(shape);
    }
    // Returns the canonical shape this variant uses at runtime. Some
    // checkpoints publish an intermediate_size in config.json that does not
    // match the actual tensor dimensions (e.g. block_auto_adjust_ff_dim). The
    // variant resolves the discrepancy here so buffer allocation and weight
    // loading see the real topology. The default returns the shape unchanged.
    virtual ModelShape resolve_shape(ModelShape shape) const { return shape; }
    // Chat template this variant must use.
    virtual ChatTemplateKind chat_template_kind() const = 0;
    // Human-readable label for diagnostics.
    virtual std::string label() const = 0;
};

// Process-wide variant registry. Variants register themselves at static-init
// time via register_variant(); the runtime selects one with select(shape).
class ModelVariantRegistry {
public:
    static ModelVariantRegistry& instance();

    void register_variant(std::unique_ptr<IModelVariant> variant);

    // Returns the matching variant or throws std::runtime_error.
    const IModelVariant& select(const ModelShape& shape) const;
    // Returns the matching variant, informed by the source repo hint captured
    // from config.json's "_name" field, or throws std::runtime_error.
    const IModelVariant& select(const ModelShape& shape,
                                std::string_view repo_hint) const;
    // Returns the variant with the given id or nullptr.
    const IModelVariant* find(std::string_view id) const;
    // All registered variant ids, in registration order.
    std::vector<std::string_view> ids() const;

private:
    ModelVariantRegistry() = default;
    std::vector<std::unique_ptr<IModelVariant>> variants_;
};

// Built-in variant: LiquidAI/LFM2.5-230M.
class Lfm25_230M_Variant final : public IModelVariant {
public:
    std::string_view id() const override;
    std::string_view repo_id() const override;
    bool matches(const ModelShape& shape) const override;
    ChatTemplateKind chat_template_kind() const override;
    std::string label() const override;
};

// Built-in variant: LiquidAI/LFM2.5-1.2B-Instruct.
class Lfm25_1_2B_Instruct_Variant final : public IModelVariant {
public:
    std::string_view id() const override;
    std::string_view repo_id() const override;
    bool matches(const ModelShape& shape) const override;
    bool matches(const ModelShape& shape, std::string_view repo_hint) const override;
    ModelShape resolve_shape(ModelShape shape) const override;
    ChatTemplateKind chat_template_kind() const override;
    std::string label() const override;
};

// Built-in variant: LiquidAI/LFM2.5-1.2B-Thinking. Shares the exact topology
// of the Instruct variant (16 layers: 10 conv + 6 GQA), so disambiguation
// relies on the config.json "_name" repo hint. Uses the same ChatML-like
// template and the same intermediate_size=8192 resolve_shape() correction.
class Lfm25_1_2B_Thinking_Variant final : public IModelVariant {
public:
    std::string_view id() const override;
    std::string_view repo_id() const override;
    bool matches(const ModelShape& shape) const override;
    bool matches(const ModelShape& shape, std::string_view repo_hint) const override;
    ModelShape resolve_shape(ModelShape shape) const override;
    ChatTemplateKind chat_template_kind() const override;
    std::string label() const override;
};

// Registers all built-in variants exactly once. Safe to call multiple times.
void register_builtin_variants();

} // namespace lfm
