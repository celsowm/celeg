#pragma once

#include "celeg/model/inference.hpp"

#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace celeg::inference_detail {

struct CanonicalInferenceContext;

[[noreturn]] void fail(ResolutionFailureKind kind, std::string message,
                       std::vector<EvidenceItem> evidence = {});

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
std::vector<std::string> norm_tensor_candidates(int layer, TensorRole role);
bool has_any_tensor(const TensorInventory& inventory,
                    const std::vector<std::string>& candidates);

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
