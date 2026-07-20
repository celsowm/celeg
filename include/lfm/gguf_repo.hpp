#pragma once

#include "lfm/gguf.hpp"
#include "lfm/safetensors.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace lfm {

// Weight repository backed by a single GGUF checkpoint file. Translates the
// canonical HuggingFace tensor names requested by the model builder
// (`model.embed_tokens.weight`, `model.layers.{i}.self_attn.q_proj.weight`, ...)
// into the llama.cpp/GGUF naming (`token_embd.weight`, `blk.{i}.attn_q.weight`,
// ...) so the model builder and weight loader remain format-agnostic.
//
// Returned HostTensorView values point directly into the memory-mapped GGUF
// file and stay valid for the lifetime of this repository. Quantized tensors
// (Q4_K/Q6_K/...) are reported with dtype == Quantized and the matching
// GgmlType; F32/F16 tensors map to the plain TensorDType values.
class GgufRepository : public IWeightRepository {
public:
    explicit GgufRepository(std::shared_ptr<GgufFile> gguf);

    bool contains(std::string_view name) const override;
    HostTensorView tensor(std::string_view name) const override;
    std::vector<std::string> names() const override;

    const GgufFile& file() const { return *gguf_; }

private:
    // Maps a canonical HF tensor name to the GGUF tensor name. Returns empty
    // string when the name has no GGUF counterpart.
    std::string translate(std::string_view hf_name) const;

    std::shared_ptr<GgufFile> gguf_;
};

} // namespace lfm
