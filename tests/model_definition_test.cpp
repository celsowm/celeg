#include "celeg/model/architecture.hpp"

#include <cassert>
#include <stdexcept>

using namespace celeg;

namespace {
ModelConfig celeg_config() {
    ModelConfig config;
    config.model_type = "lfm2";
    config.hidden_size = 8;
    config.intermediate_size = 16;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.head_dim = 4;
    config.vocab_size = 32;
    config.max_position_embeddings = 128;
    config.norm_eps = 1e-5f;
    config.rope_theta = 10000.0f;
    config.use_pos_enc = true;
    return config;
}

ModelConfig granite_config() {
    ModelConfig config = celeg_config();
    config.model_type = "granite";
    config.architecture = ArchitectureKind::Granite;
    config.embedding_multiplier = 2.0f;
    config.attention_multiplier = 0.125f;
    config.residual_multiplier = 0.5f;
    config.logits_divisor = 2.0f;
    config.query_key_norm = false;
    config.layer_types.assign(2, LayerType::FullAttention);
    return config;
}
}

int main() {
    register_builtin_architecture_providers();
    const auto& registry = ArchitectureRegistry::instance();
    const auto& provider = registry.select(celeg_config());
    assert(provider.id() == "lfm2");
    const ModelDefinition definition = provider.inspect(celeg_config());
    assert(definition.dimensions.hidden_size == 8);
    assert(definition.rope.kind == PositionalEncodingKind::Rope);

    const auto& granite_provider = registry.select(granite_config());
    assert(granite_provider.id() == "granite");
    const ModelDefinition granite = granite_provider.inspect(granite_config());
    assert(granite.numerics.embedding_multiplier == 2.0f);
    assert(granite.numerics.logits_divisor == 2.0f);
    assert(granite_provider.tensor_naming().candidates(
        {TensorRole::AttentionOutput, 0, -1, {}}).front() ==
        "model.layers.0.self_attn.o_proj.weight");

    ModelConfig unknown = celeg_config();
    unknown.model_type = "unknown";
    unknown.architecture = static_cast<ArchitectureKind>(99);
    bool rejected = false;
    try { (void)registry.select(unknown); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    return 0;
}
