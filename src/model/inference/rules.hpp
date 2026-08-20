#pragma once

#include "canonical_internal.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace celeg::inference_detail {

/// Coarse mixer family used for mutual-exclusion conflict reporting: a
/// layer cannot simultaneously carry an attention grammar and a recurrent
/// grammar, regardless of which rule would win the specificity contest.
enum class MixerFamily : uint8_t {
    Attention,
    Recurrent,
    Convolutional,
    PureMlp,
};

/// A load-time tensor-grammar rule for one mixer family. A rule owns the
/// whole recognition contract for its family: the grammar probe, the
/// semantic mixer it produces, and the tensor-role bindings that justify
/// it, so semantic inference and binding dispatch can no longer drift
/// apart. Grammar recognition is open for extension (compose another rule
/// into the catalog); the semantic MixerSpec alternatives remain
/// deliberately closed and compile-time exhaustive.
class ILayerInferenceRule {
public:
    virtual ~ILayerInferenceRule() = default;

    /// Stable rule identifier used in diagnostics.
    virtual std::string_view id() const = 0;

    /// Grammar precedence: when several rules probe-match a layer, the
    /// highest specificity resolves it, and two matches at equal
    /// specificity are a reported conflict rather than an accidental
    /// registration order.
    virtual int specificity() const = 0;

    /// Family tag driving mutual-exclusion conflict reporting.
    virtual MixerFamily family() const = 0;

    /// Reports whether the layer's inventory carries this rule's grammar.
    /// Must stay side-effect free.
    virtual bool probe(const CanonicalInferenceContext& context,
                       int layer) const = 0;

    /// Emits the semantic mixer and the tensor-role bindings that justify
    /// it for one matched layer, including clearing the feed-forward axis
    /// to monostate when the layer carries no feed-forward grammar. Called
    /// only after the catalog selected this rule.
    virtual void resolve(CanonicalInferenceContext& context,
                         int layer) const = 0;
};

/// Constructs the built-in rule catalog. Callers own the returned catalog
/// so extension code and tests can compose it with additional rules before
/// selection.
std::vector<std::unique_ptr<ILayerInferenceRule>>
make_builtin_layer_inference_rules();

std::unique_ptr<ILayerInferenceRule> make_standard_attention_rule();
std::unique_ptr<ILayerInferenceRule> make_latent_attention_rule();
std::unique_ptr<ILayerInferenceRule> make_fused_gated_delta_rule();
std::unique_ptr<ILayerInferenceRule> make_linear_attn_gated_delta_rule();
std::unique_ptr<ILayerInferenceRule> make_factorized_gated_delta_rule();
std::unique_ptr<ILayerInferenceRule> make_mamba2_rule();
std::unique_ptr<ILayerInferenceRule> make_short_convolution_rule();
std::unique_ptr<ILayerInferenceRule> make_mlp_only_rule();

/// Probes every rule against one layer and returns the rule that wins by
/// specificity. Fails with an explicit conflict when two rules match at
/// the same specificity or when attention and recurrent grammars coexist
/// on one layer, and fails when no rule recognizes the layer at all.
const ILayerInferenceRule& select_layer_inference_rule(
    std::span<const std::unique_ptr<ILayerInferenceRule>> rules,
    const CanonicalInferenceContext& context,
    int layer);

}
