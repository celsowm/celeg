#include "celeg/model/architecture.hpp"

#include <algorithm>
#include <stdexcept>

namespace celeg {
namespace {

const CelegTensorNamingPolicy& celeg_tensor_naming() {
    static const CelegTensorNamingPolicy policy;
    return policy;
}

const GraniteTensorNamingPolicy& granite_tensor_naming() {
    static const GraniteTensorNamingPolicy policy;
    return policy;
}

ModelDefinition common_definition(const ModelConfig& config,
                                  std::string_view architecture) {
    ModelDefinition definition;
    definition.dimensions = {
        config.hidden_size, config.intermediate_size, config.num_hidden_layers,
        config.num_attention_heads, config.num_key_value_heads, config.head_dim,
        config.vocab_size, config.max_position_embeddings};
    definition.rope.kind = config.use_pos_enc
        ? PositionalEncodingKind::Rope : PositionalEncodingKind::None;
    definition.rope.theta = config.rope_theta;
    definition.numerics.norm_epsilon = config.norm_eps;
    definition.numerics.embedding_multiplier = config.embedding_multiplier;
    definition.numerics.attention_multiplier = config.attention_multiplier;
    definition.numerics.residual_multiplier = config.residual_multiplier;
    definition.numerics.logits_divisor = config.logits_divisor;
    definition.tokens = {config.bos_token_id, config.eos_token_id, config.pad_token_id};
    definition.architecture = std::string(architecture);
    definition.source_format = config.is_gguf ? "gguf" : "safetensors";
    definition.validate();
    return definition;
}

}

ArchitectureRegistry& ArchitectureRegistry::instance() {
    static ArchitectureRegistry registry;
    return registry;
}

void ArchitectureRegistry::register_provider(
    std::unique_ptr<IArchitectureProvider> provider) {
    if (!provider || provider->id().empty()) {
        throw std::invalid_argument("architecture provider must have an id");
    }
    if (find(provider->id()) != nullptr) {
        throw std::invalid_argument("duplicate architecture provider: " +
                                    std::string(provider->id()));
    }
    providers_.push_back(std::move(provider));
}

const IArchitectureProvider& ArchitectureRegistry::select(
    const ModelConfig& config) const {
    const IArchitectureProvider* selected = nullptr;
    for (const auto& provider : providers_) {
        if (!provider->supports(config)) continue;
        if (selected != nullptr) {
            throw std::runtime_error("multiple architecture providers match checkpoint");
        }
        selected = provider.get();
    }
    if (selected == nullptr) {
        throw std::runtime_error("no architecture provider supports checkpoint model_type: " +
                                 config.model_type);
    }
    return *selected;
}

const IArchitectureProvider* ArchitectureRegistry::find(std::string_view id) const {
    const auto it = std::find_if(providers_.begin(), providers_.end(),
        [id](const auto& provider) { return provider->id() == id; });
    return it == providers_.end() ? nullptr : it->get();
}

std::vector<std::string_view> ArchitectureRegistry::ids() const {
    std::vector<std::string_view> result;
    result.reserve(providers_.size());
    for (const auto& provider : providers_) result.push_back(provider->id());
    return result;
}

std::string_view CelegArchitectureProvider::id() const { return "lfm2"; }

bool CelegArchitectureProvider::supports(const ModelConfig& config) const {
    return config.model_type == "lfm2" || config.model_type == "lfm2_moe" ||
           config.architecture == ArchitectureKind::DenseLfm2 ||
           config.architecture == ArchitectureKind::MoeLfm2;
}

ModelDefinition CelegArchitectureProvider::inspect(const ModelConfig& config) const {
    return common_definition(config, id());
}

const ITensorNamingPolicy& CelegArchitectureProvider::tensor_naming() const {
    return celeg_tensor_naming();
}

std::string_view GraniteArchitectureProvider::id() const { return "granite"; }

bool GraniteArchitectureProvider::supports(const ModelConfig& config) const {
    return config.architecture == ArchitectureKind::Granite;
}

ModelDefinition GraniteArchitectureProvider::inspect(const ModelConfig& config) const {
    return common_definition(config, id());
}

const ITensorNamingPolicy& GraniteArchitectureProvider::tensor_naming() const {
    return granite_tensor_naming();
}

void register_builtin_architecture_providers() {
    auto& registry = ArchitectureRegistry::instance();
    if (registry.find("lfm2") == nullptr)
        registry.register_provider(std::make_unique<CelegArchitectureProvider>());
    if (registry.find("granite") == nullptr)
        registry.register_provider(std::make_unique<GraniteArchitectureProvider>());
}

} // namespace celeg
