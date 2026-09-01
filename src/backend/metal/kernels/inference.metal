/**
 * @file inference.metal
 * @brief Translation unit index for the Metal inference kernel families.
 *
 * The runtime source is assembled by CMake in this same order. Keeping the
 * includes here also lets Metal tooling inspect the complete shader locally.
 */
#include "inference/common.metal"
#include "inference/vector.metal"
#include "inference/state.metal"
#include "inference/projection.metal"
#include "inference/batch.metal"
#include "inference/pair.metal"
#include "inference/relative_bias.metal"
#include "inference/position.metal"
#include "inference/no_position.metal"
#include "inference/qk_position.metal"
#include "inference/qk_norm.metal"
#include "inference/mrope_batch.metal"
#include "inference/attention_output.metal"
#include "inference/attention_gate.metal"
