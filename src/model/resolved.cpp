#include "celeg/model/resolved.hpp"

#include <sstream>
#include <stdexcept>

namespace celeg {

std::string RuntimeTopology::fingerprint() const {
    std::ostringstream out;
    out << "h" << hidden << "-l" << num_hidden_layers
        << "-qh" << num_attention_heads << "-kvh" << num_key_value_heads
        << "-hd" << head_dim << "-voc" << vocab_size
        << "-int" << intermediate << "-cc" << conv_cache
        << "-e" << num_experts << "-k" << experts_per_token
        << "-mi" << moe_intermediate;
    for (MixerKind kind : mixer_kinds) out << (kind == MixerKind::Attention ? "-a" : "-c");
    return out.str();
}

std::string RuntimeTopology::summary() const {
    std::ostringstream out;
    out << "hidden=" << hidden << " intermediate=" << intermediate
        << " layers=" << num_hidden_layers
        << " attention_layers=" << attention_layer_count
        << " conv_layers=" << conv_layer_count
        << " q_heads=" << num_attention_heads
        << " kv_heads=" << num_key_value_heads
        << " head_dim=" << head_dim << " vocab=" << vocab_size;
    return out.str();
}

void RuntimeTopology::validate() const {
    if (hidden <= 0 || intermediate <= 0 || vocab_size <= 0 ||
        num_hidden_layers <= 0 || num_attention_heads <= 0 ||
        num_key_value_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("invalid resolved model topology");
    }
    if (num_attention_heads % num_key_value_heads != 0 ||
        hidden != num_attention_heads * head_dim || (head_dim % 2) != 0) {
        throw std::runtime_error("invalid resolved attention topology");
    }
    if (static_cast<int>(mixer_kinds.size()) != num_hidden_layers ||
        static_cast<int>(layer_types.size()) != num_hidden_layers) {
        throw std::runtime_error("resolved mixer schedule length mismatch: mixers=" +
            std::to_string(mixer_kinds.size()) + " layers=" +
            std::to_string(layer_types.size()) + " expected=" +
            std::to_string(num_hidden_layers));
    }
    if (attention_layer_count + conv_layer_count != num_hidden_layers) {
        throw std::runtime_error("resolved layer counts are inconsistent");
    }
    if (num_experts > 0 && (moe_intermediate <= 0 || experts_per_token <= 0 ||
                            experts_per_token > num_experts)) {
        throw std::runtime_error("invalid resolved MoE topology");
    }
    if (q_width != num_attention_heads * head_dim ||
        kv_width != num_key_value_heads * head_dim ||
        qkv_width != q_width + 2 * kv_width || rope_pairs != head_dim / 2) {
        throw std::runtime_error("invalid derived resolved dimensions");
    }
}

} // namespace celeg
