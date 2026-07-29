#include "lfm/model/config/shape.hpp"

#include <sstream>
#include <stdexcept>

namespace lfm {

ModelShape ModelShape::from_config(const ModelConfig& config) {
    ModelShape shape;
    shape.hidden = config.hidden_size;
    shape.intermediate = config.intermediate_size;
    shape.num_hidden_layers = config.num_hidden_layers;
    shape.num_attention_heads = config.num_attention_heads;
    shape.num_key_value_heads = config.num_key_value_heads;
    shape.head_dim = config.head_dim;
    shape.vocab_size = config.vocab_size;
    shape.conv_cache = config.conv_cache;
    shape.conv_dim = config.conv_dim;
    shape.max_position_embeddings = config.max_position_embeddings;
    shape.bos_token_id = config.bos_token_id;
    shape.eos_token_id = config.eos_token_id;
    shape.pad_token_id = config.pad_token_id;
    shape.norm_eps = config.norm_eps;
    shape.rope_theta = config.rope_theta;
    shape.rope_type = config.rope_type;
    shape.layer_types = config.layer_types;
    shape.architecture = config.architecture;
    if (config.architecture == ArchitectureKind::MoeLfm2 && config.moe) {
        const MoeConfig& m = *config.moe;
        shape.dense_intermediate = m.intermediate_size;
        shape.moe_intermediate = m.moe_intermediate_size;
        shape.num_dense_layers = m.num_dense_layers;
        shape.num_experts = m.num_experts;
        shape.experts_per_token = m.experts_per_token;
        shape.normalize_topk = m.normalize_topk;
        shape.use_expert_bias = m.use_expert_bias;
        shape.routed_scaling_factor = m.routed_scaling_factor;
    }
    shape.compute_derived();
    shape.validate();
    return shape;
}

void ModelShape::compute_derived() {
    q_width = num_attention_heads * head_dim;
    kv_width = num_key_value_heads * head_dim;
    qkv_width = q_width + 2 * kv_width;
    rope_pairs = head_dim / 2;
    attention_layer_count = 0;
    conv_layer_count = 0;
    attention_slot_for_layer.assign(static_cast<size_t>(num_hidden_layers), -1);
    layer_for_attention_slot.clear();
    layer_for_attention_slot.reserve(static_cast<size_t>(num_hidden_layers));
    for (int index = 0; index < num_hidden_layers; ++index) {
        const LayerType type = layer_types[static_cast<size_t>(index)];
        if (type == LayerType::FullAttention) {
            attention_slot_for_layer[static_cast<size_t>(index)] =
                static_cast<int>(layer_for_attention_slot.size());
            layer_for_attention_slot.push_back(index);
            ++attention_layer_count;
        } else {
            ++conv_layer_count;
        }
    }
}

void ModelShape::validate() const {
    if (hidden <= 0 || intermediate <= 0 || vocab_size <= 0) {
        throw std::runtime_error("invalid non-positive model dimensions");
    }
    if (architecture == ArchitectureKind::MoeLfm2) {
        if (moe_intermediate <= 0) {
            throw std::runtime_error("invalid MoE expert intermediate size");
        }
        if (num_experts <= 0) throw std::runtime_error("invalid MoE expert count");
        if (experts_per_token <= 0) {
            throw std::runtime_error("invalid MoE experts_per_token");
        }
        if (experts_per_token > num_experts) {
            throw std::runtime_error("experts_per_token exceeds num_experts");
        }
        if (num_dense_layers < 0) {
            throw std::runtime_error("invalid MoE dense layer count");
        }
        if (num_dense_layers > num_hidden_layers) {
            throw std::runtime_error("MoE dense layer count exceeds total layers");
        }
    }
    if (num_hidden_layers <= 0 ||
        static_cast<int>(layer_types.size()) != num_hidden_layers) {
        throw std::runtime_error("layer_types length does not match num_hidden_layers");
    }
    if (num_attention_heads <= 0 || num_key_value_heads <= 0 ||
        num_attention_heads % num_key_value_heads != 0) {
        throw std::runtime_error("invalid GQA head configuration");
    }
    if (head_dim <= 0 || head_dim * num_attention_heads != hidden || (head_dim % 2) != 0) {
        throw std::runtime_error("invalid attention head_dim");
    }
    if (conv_dim != hidden || conv_cache <= 0) {
        throw std::runtime_error("unsupported convolution dimensions");
    }
    if (max_position_embeddings <= 0) {
        throw std::runtime_error("invalid max_position_embeddings");
    }
    if (q_width != num_attention_heads * head_dim ||
        kv_width != num_key_value_heads * head_dim ||
        qkv_width != q_width + 2 * kv_width ||
        rope_pairs != head_dim / 2) {
        throw std::runtime_error("model shape derived fields are inconsistent");
    }
    if (attention_layer_count + conv_layer_count != num_hidden_layers) {
        throw std::runtime_error("attention/conv layer counts are inconsistent");
    }
    if (static_cast<int>(layer_for_attention_slot.size()) != attention_layer_count) {
        throw std::runtime_error("attention layer slot table is inconsistent");
    }
}

std::string ModelShape::fingerprint() const {
    std::ostringstream out;
    out << "h" << hidden
        << "-l" << num_hidden_layers
        << "-qh" << num_attention_heads
        << "-kvh" << num_key_value_heads
        << "-hd" << head_dim
        << "-voc" << vocab_size
        << "-int" << intermediate
        << "-cc" << conv_cache;
    if (architecture == ArchitectureKind::MoeLfm2) {
        out << "-moe"
            << "-e" << num_experts
            << "-k" << experts_per_token
            << "-d" << num_dense_layers
            << "-mi" << moe_intermediate
            << "-nt" << (normalize_topk ? 1 : 0)
            << "-eb" << (use_expert_bias ? 1 : 0)
            << "-rs" << static_cast<int>(routed_scaling_factor);
    }
    return out.str();
}

std::string ModelShape::summary() const {
    std::ostringstream out;
    out << "hidden=" << hidden
        << " intermediate=" << intermediate
        << " layers=" << num_hidden_layers
        << " attention_layers=" << attention_layer_count
        << " conv_layers=" << conv_layer_count
        << " q_heads=" << num_attention_heads
        << " kv_heads=" << num_key_value_heads
        << " head_dim=" << head_dim
        << " vocab=" << vocab_size
        << " max_positions=" << max_position_embeddings
        << " rope_theta=" << rope_theta;
    if (architecture == ArchitectureKind::MoeLfm2) {
        out << " arch=moe"
            << " dense_intermediate=" << dense_intermediate
            << " moe_intermediate=" << moe_intermediate
            << " num_dense_layers=" << num_dense_layers
            << " num_experts=" << num_experts
            << " experts_per_token=" << experts_per_token
            << " normalize_topk=" << (normalize_topk ? 1 : 0)
            << " use_expert_bias=" << (use_expert_bias ? 1 : 0)
            << " routed_scaling_factor=" << routed_scaling_factor;
    }
    return out.str();
}

} // namespace lfm
