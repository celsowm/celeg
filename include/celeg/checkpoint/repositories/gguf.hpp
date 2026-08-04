#pragma once

#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/formats/safetensors.hpp"
#include "celeg/checkpoint/tokenizer.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace celeg {

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
class GgufRepository : public IWeightRepository,
                       public INativeBlockStorageRepository,
                       public ITokenizerDataRepository {
public:
    explicit GgufRepository(std::shared_ptr<GgufFile> gguf);

    bool contains(std::string_view name) const override;
    HostTensorView tensor(std::string_view name) const override;
    std::vector<std::string> names() const override;
    bool has_native_block_storage() const override { return true; }
    TokenizerData tokenizer_data() const override;

    const GgufFile& file() const { return *gguf_; }

private:
    // Maps a canonical HF tensor name to the GGUF tensor name. Returns empty
    // string when the name has no GGUF counterpart.
    std::string translate(std::string_view hf_name) const;

    std::shared_ptr<GgufFile> gguf_;
};

} // namespace celeg
