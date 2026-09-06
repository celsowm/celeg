#pragma once

#include "celeg/model/inference.hpp"

#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace celeg::inference_detail {

struct CanonicalInferenceContext;

[[noreturn]] void fail(ResolutionFailureKind kind, std::string message,
                       std::vector<EvidenceItem> evidence = {});

/// Consumption ledger for the unknown-semantics gate: every metadata helper
/// records the exact key it successfully read, so `reject_unknown_semantic_metadata`
/// can report semantic keys the resolver never consumed instead of silently
/// dropping them. Warns by default, fails under `CELEG_STRICT_SEMANTICS=1`.
void record_metadata_consumption(std::string_view key);
void clear_metadata_consumption();
const std::unordered_set<std::string>& metadata_consumption();
void reject_unknown_semantic_metadata(const CheckpointMetadata& metadata,
                                      const TensorInventory& inventory);

/// True when a metadata value is provably inert (false, integer zero, or
/// floating zero): opt-in bias/norm flags with such a value enable no extra
/// mathematics, so the unknown-semantics gate may ignore exactly that case
/// while still failing loudly when the flag is set.
bool metadata_value_is_falsy(const MetadataValue& value);

bool shape_is(const TensorInventoryEntry& entry,
              std::initializer_list<std::int64_t> expected);

const TensorInventoryEntry* find_unique(const TensorInventory& inventory,
                                        const std::vector<std::string>& candidates,
                                        TensorRole role, int layer,
                                        std::initializer_list<std::int64_t> shape,
                                        std::vector<EvidenceItem> evidence);

std::vector<std::string> attention_tensor_candidates(int layer,
                                                      std::string_view suffix);
std::vector<std::string> feed_forward_tensor_candidates(int layer,
                                                        std::string_view suffix);
std::vector<std::string> shortconv_tensor_candidates(int layer,
                                                     std::string_view suffix);
std::vector<std::string> mamba2_tensor_candidates(int layer,
                                                  std::string_view suffix);

const TensorInventoryEntry* find_mamba_tensor(const InferenceInput& input,
                                              int layer,
                                              std::string_view suffix);

/// Whether the layer's inventory carries any feed-forward grammar (dense
/// projections in any known convention, or the first routed-expert tensor of
/// a checkpoint MoE family). Shared by the mixer rules (absence turns the
/// layer feed-forward into monostate) and the MLP-only grammar rule.
bool layer_has_feed_forward(const CanonicalInferenceContext& context,
                            int layer);

void add_binding(TensorRoleBindings& bindings, TensorRole role, int layer,
                 const TensorInventoryEntry& tensor,
                 std::vector<EvidenceItem> evidence, int physical_layer = -1);

}
