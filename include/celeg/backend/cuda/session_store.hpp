#pragma once

#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "celeg/backend/cuda/utils.cuh"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace celeg {

// Persistence boundary for a CudaModel session. Isolates the on-disk
// session-file format (header layout, byte order, payload ordering, magic
// number, version, resolved-model fingerprint) and the prefix-state snapshot
// representation from the inference path (Single Responsibility Principle).
// New file-format versions or new prefix-state representations are handled
// at this persistence boundary without touching the forward pass.
//
// The store exposes only static helpers: it never retains ownership and
// never holds a reference to the host. The compiled model passes
// the live device buffers and topology snapshot per call.
class SessionStore {
public:
    // Header layout for the v4 session file. Bumping the version requires
    // changing the magic and the version field; old files are rejected
    // cleanly without a migration path because the on-disk layout now
    // depends on the resolved-model fingerprint.
    struct Header {
        std::array<char, 8> magic{{'C', 'E', 'L', 'E', 'G', 'S', 'S', '4'}};
        uint32_t version = 4;
        uint32_t kv_cache_mode = 0;
        int32_t position = 0;
        int32_t max_context = 0;
        int32_t layers = 0;
        int32_t vocab = 0;
        int32_t attention_layers = 0;
        uint64_t rng_state = 0;
        char model_identity[96] = {};
    };

    static constexpr std::array<char, 8> kMagic{
        {'C', 'E', 'L', 'E', 'G', 'S', 'S', '4'}};
    static constexpr uint32_t kVersion = 4;

    // Snapshot of the live session state that SessionStore reads from or
    // writes to. The host fills this struct and passes it in; the store
    // never retains a reference beyond the call.
    struct SessionState {
        const ExecutionTopology& shape;
        const CompiledModelProgram& program;
        const CheckpointDimensions& dims;
        int max_context = 0;
        int position = 0;
        KvCacheMode kv_cache_mode = KvCacheMode::Bf16;
        std::string model_identity;
        cudaStream_t stream = nullptr;
        DeviceBuffer<uint8_t>* seen_tokens = nullptr;
        DeviceBuffer<__nv_bfloat16>* logits = nullptr;
        DeviceBuffer<uint64_t>* rng_state = nullptr;
        // Per-layer K/V + conv-state buffers. The host populates this with
        // pointers into the live AttentionLayer / ConvolutionLayer storage.
        struct LayerBuffers {
            bool is_attention = false;
            bool is_mamba = false;
            bool is_gated_delta = false;
            bool owns_kv_cache = false;
            // BF16 path (when kv_cache_mode == Bf16):
            __nv_bfloat16* key_cache_bf16 = nullptr;
            __nv_bfloat16* value_cache_bf16 = nullptr;
            // INT8 path (when kv_cache_mode == Int8):
            int8_t* key_cache_int8 = nullptr;
            int8_t* value_cache_int8 = nullptr;
            float* key_cache_scales = nullptr;
            float* value_cache_scales = nullptr;
            // Convolution path:
            __nv_bfloat16* conv_state = nullptr;
            size_t conv_state_elements = 0;
            __nv_bfloat16* ssm_state = nullptr;
            size_t ssm_state_elements = 0;
            __nv_bfloat16* recurrent_state = nullptr;
            size_t recurrent_state_elements = 0;
        };
        std::vector<LayerBuffers> layer_buffers;
    };

    // Writes a v3 session file at `path`. Throws on any I/O or CUDA error.
    static void save(const std::string& path, SessionState& state);

    // Reads a v3 session file from `path` and restores the buffers. Throws
    // if the file is not a v3 session, the dimensions mismatch, or the
        // resolved-model fingerprint differs. On success, `state.position` is updated
    // to match the header.
    static void load(const std::string& path, SessionState& state);

    // Snapshot of the prefix-state-of-record for cache reuse. Host-owned
    // bytes; SessionStore only copies. Uses uint16_t for the BF16 payloads
    // (bitwise-compatible with __nv_bfloat16) so the snapshot moves into
    // the public PrefixState struct without conversion.
    struct PrefixSnapshot {
        int position = 0;
        std::vector<uint8_t> seen_tokens;
        std::vector<uint16_t> logits_bf16;
        std::vector<uint16_t> conv_state_bf16;
        std::vector<uint16_t> mamba_state_bf16;
        std::vector<uint16_t> gated_delta_state_bf16;
    };

    // Exports a prefix snapshot (host-side copies of seen_tokens, logits,
    // and the conv_state of every convolution layer).
    static PrefixSnapshot export_prefix(const SessionState& state);

    // Restores a prefix snapshot. Resets phase to Ready on success; the
    // RNG is NOT inherited from the snapshot — `request_seed` is written
    // instead, so the receiving request keeps its own generation stream.
    // Updates `state.position` to the snapshot's position.
    static void restore_prefix(const PrefixSnapshot& snapshot,
                               SessionState& state,
                               uint64_t request_seed);
};

} // namespace celeg
