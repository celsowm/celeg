#include "backend/cuda/attention_capability.hpp"
#include "backend/cuda/compiler.hpp"
#include "support/assertions.hpp"

#include <stdexcept>

namespace {

celeg::ResolvedModel relative_model(bool bidirectional = false) {
    celeg::ResolvedModel model;
    model.provenance.identity = "cuda-relative-bias-fixture";
    model.graph.hidden = 8;

    celeg::LayerSpec layer;
    layer.mixer_norm.before = celeg::NormSpec{
        1.0e-5f, celeg::NormWeightKind::Scale};
    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = 8;
    attention.position = celeg::NoPositionEncodingSpec{};
    attention.bias = celeg::RelativePositionBiasSpec{
        32, 128, bidirectional};
    layer.mixer = attention;
    layer.feed_forward = std::monostate{};
    model.graph.layers.push_back(std::move(layer));

    for (celeg::TensorRole role : {
             celeg::TensorRole::AttentionInputNorm,
             celeg::TensorRole::AttentionQuery,
             celeg::TensorRole::AttentionKey,
             celeg::TensorRole::AttentionValue,
             celeg::TensorRole::AttentionOutput,
             celeg::TensorRole::AttentionRelativePositionBias}) {
        model.weight_plan.requests.push_back({role, 0, -1, {}});
    }
    return model;
}

template <typename Mutator>
bool compiler_rejects(Mutator mutate) {
    celeg::ResolvedModel model = relative_model();
    mutate(std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer));
    try {
        (void)celeg::CudaModelCompiler{}.compile(model);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void physical_registry_supports_relative_bias() {
    for (celeg::KvCacheMode format :
         {celeg::KvCacheMode::Bf16, celeg::KvCacheMode::Int8}) {
        const auto prefill = celeg::attention_capability(
            format, celeg::AttentionPositionBias::Relative,
            celeg::AttentionOperation::Prefill,
            celeg::AttentionKvLayout::Contiguous,
            celeg::AttentionPositionSource::HostScalar,
            celeg::AttentionAlgorithm::RelativeBias);
        CELEG_TEST_CHECK(prefill.supported);

        for (celeg::AttentionKvLayout layout :
             {celeg::AttentionKvLayout::Contiguous,
              celeg::AttentionKvLayout::Paged,
              celeg::AttentionKvLayout::BatchPointers}) {
            const auto decode = celeg::attention_capability(
                format, celeg::AttentionPositionBias::Relative,
                celeg::AttentionOperation::Decode, layout,
                celeg::AttentionPositionSource::DeviceCounter,
                celeg::AttentionAlgorithm::RelativeBias);
            CELEG_TEST_CHECK(decode.supported);
        }

        const auto host_decode = celeg::attention_capability(
            format, celeg::AttentionPositionBias::Relative,
            celeg::AttentionOperation::Decode,
            celeg::AttentionKvLayout::Contiguous,
            celeg::AttentionPositionSource::HostScalar,
            celeg::AttentionAlgorithm::RelativeBias);
        CELEG_TEST_CHECK(!host_decode.supported);
        CELEG_TEST_CHECK(host_decode.reason ==
            celeg::AttentionUnsupportedReason::PositionSourceNotImplemented);
    }
}

void compiler_accepts_standard_relative_bias() {
    const celeg::CompiledModelProgram causal =
        celeg::CudaModelCompiler{}.compile(relative_model(false));
    CELEG_TEST_CHECK(causal.layers.size() == 1);
    const auto& causal_attention = std::get<celeg::CompiledAttentionProgram>(
        causal.layers[0].mixer);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::RelativePositionBiasSpec>(
        causal_attention.semantics.bias));
    CELEG_TEST_CHECK(causal_attention.execution.kind ==
        celeg::AttentionExecutionKind::Standard);

    celeg::ResolvedModel bidirectional = relative_model(true);
    auto& bidirectional_attention = std::get<celeg::AttentionSpec>(
        bidirectional.graph.layers[0].mixer);
    bidirectional_attention.pattern = celeg::BidirectionalPattern{};
    const celeg::CompiledModelProgram compiled_bidirectional =
        celeg::CudaModelCompiler{}.compile(bidirectional);
    const auto& compiled_attention = std::get<celeg::CompiledAttentionProgram>(
        compiled_bidirectional.layers[0].mixer);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::BidirectionalPattern>(
        compiled_attention.semantics.pattern));
    const auto& relative = std::get<celeg::RelativePositionBiasSpec>(
        compiled_attention.semantics.bias);
    CELEG_TEST_CHECK(relative.bidirectional);
}

void unsupported_relative_semantics_are_explicit() {
    CELEG_TEST_CHECK(compiler_rejects([](auto& attention) {
        attention.pattern = celeg::PrefixLmPattern{4};
    }));

    CELEG_TEST_CHECK(compiler_rejects([](auto& attention) {
        attention.pattern = celeg::BlockSparsePattern{16, 2, 1};
    }));

    CELEG_TEST_CHECK(compiler_rejects([](auto& attention) {
        attention.state = celeg::LatentAttentionStateSpec{16, 8, 8, true};
    }));
}

}

int main() {
    physical_registry_supports_relative_bias();
    compiler_accepts_standard_relative_bias();
    unsupported_relative_semantics_are_explicit();
    return 0;
}
