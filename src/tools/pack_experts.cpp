#include "lfm/checkpoint/repositories/safetensors.hpp"
#include "lfm/model/config/config.hpp"
#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/detail/binary_codec.hpp"
#include "lfm/model/config/shape.hpp"
#include "lfm/runtime/moe/offload.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace {

struct SidecarHeader {
    char magic[8] = {'L', 'F', 'M', 'S', 'I', 'D', 'E', '2'};
    std::uint32_t num_layers = 0;
    std::uint32_t num_experts = 0;
    std::uint64_t moe_intermediate = 0;
    std::uint64_t hidden = 0;
    std::uint64_t reserved[4] = {0, 0, 0, 0};
};

struct SidecarExpertIndex {
    std::uint64_t gate_up_offset = 0;
    std::uint64_t gate_up_bytes = 0;
    std::uint64_t down_offset = 0;
    std::uint64_t down_bytes = 0;
};

void write_header(std::ostream& output, const SidecarHeader& header) {
    output.write(header.magic, sizeof(header.magic));
    lfm::binary::write_le(output, header.num_layers);
    lfm::binary::write_le(output, header.num_experts);
    lfm::binary::write_le(output, header.moe_intermediate);
    lfm::binary::write_le(output, header.hidden);
    for (const std::uint64_t value : header.reserved) {
        lfm::binary::write_le(output, value);
    }
}

void write_index(std::ostream& output, const SidecarExpertIndex& index) {
    lfm::binary::write_le(output, index.gate_up_offset);
    lfm::binary::write_le(output, index.gate_up_bytes);
    lfm::binary::write_le(output, index.down_offset);
    lfm::binary::write_le(output, index.down_bytes);
}

std::string layer_name(int layer, const std::string& suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: lfm25-pack-experts SAFETENSORS_PATH OUTPUT_SIDECAR_PATH\n";
        return 2;
    }

    try {
        std::filesystem::path model_path(argv[1]);
        std::filesystem::path sidecar_path(argv[2]);

        const lfm::detail::ModelBootstrap bootstrap =
            lfm::detail::load_model_bootstrap(model_path);
        const lfm::ModelConfig& config = bootstrap.config;
        const lfm::ModelShape& shape = bootstrap.shape;

        int moe_layers = lfm::moe_layer_count(shape);
        if (moe_layers == 0) {
            std::cout << "No MoE layers in this model variant, nothing to pack.\n";
            return 0;
        }

        std::cout << "Packing MoE experts for " << config.repo_hint << "...\n"
                  << "  MoE layers: " << moe_layers << "\n"
                  << "  Experts/layer: " << shape.num_experts << "\n"
                  << "  Intermediate: " << shape.moe_intermediate << "\n"
                  << "  Hidden: " << shape.hidden << "\n";

        lfm::SafeTensorRepository repo(model_path);

        SidecarHeader header;
        header.num_layers = static_cast<std::uint32_t>(moe_layers);
        header.num_experts = static_cast<std::uint32_t>(shape.num_experts);
        header.moe_intermediate = static_cast<std::uint64_t>(shape.moe_intermediate);
        header.hidden = static_cast<std::uint64_t>(shape.hidden);

        std::vector<std::vector<SidecarExpertIndex>> index(
            moe_layers, std::vector<SidecarExpertIndex>(shape.num_experts));

        std::ofstream out(sidecar_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("cannot open output sidecar file: " + sidecar_path.string());
        }

        // Write placeholder header
        write_header(out, header);

        // Write placeholder index
        for (int l = 0; l < moe_layers; ++l) {
            for (const SidecarExpertIndex& entry : index[l]) write_index(out, entry);
        }

        // Align start of data to 4096 bytes
        std::uint64_t index_end_pos = out.tellp();
        std::uint64_t aligned_start = (index_end_pos + 4095) & ~4095ull;
        std::vector<char> padding(aligned_start - index_end_pos, 0);
        if (!padding.empty()) {
            out.write(padding.data(), padding.size());
        }

        const size_t moe_inter = static_cast<size_t>(shape.moe_intermediate);
        const size_t hidden_c = static_cast<size_t>(shape.hidden);
        const size_t gate_up_elems = 2 * moe_inter * hidden_c;
        const size_t down_elems = hidden_c * moe_inter;
        const size_t gate_up_bytes = gate_up_elems * 2; // BF16 element size = 2
        const size_t down_bytes = down_elems * 2;
        const size_t w_bytes = moe_inter * hidden_c * 2;

        std::vector<char> gate_up_stage(gate_up_bytes);
        std::vector<char> down_stage(down_bytes);

        for (int l = 0; l < moe_layers; ++l) {
            int actual_layer_idx = shape.num_dense_layers + l;
            std::cout << "Packing layer " << actual_layer_idx << " (" << (l + 1) << "/" << moe_layers << ")...\n";

            for (int e = 0; e < shape.num_experts; ++e) {
                const std::string w1_name = layer_name(
                    actual_layer_idx, "feed_forward.experts." + std::to_string(e) + ".w1.weight");
                const std::string w3_name = layer_name(
                    actual_layer_idx, "feed_forward.experts." + std::to_string(e) + ".w3.weight");
                const std::string w2_name = layer_name(
                    actual_layer_idx, "feed_forward.experts." + std::to_string(e) + ".w2.weight");

                lfm::HostTensorView w1 = repo.tensor(w1_name);
                lfm::HostTensorView w3 = repo.tensor(w3_name);
                lfm::HostTensorView w2 = repo.tensor(w2_name);

                std::memcpy(gate_up_stage.data(), w1.data, w_bytes);
                std::memcpy(gate_up_stage.data() + w_bytes, w3.data, w_bytes);
                std::memcpy(down_stage.data(), w2.data, down_bytes);

                // 4 KiB align each expert block
                std::uint64_t current_pos = out.tellp();
                std::uint64_t aligned_block_start = (current_pos + 4095) & ~4095ull;
                std::vector<char> block_pad(aligned_block_start - current_pos, 0);
                if (!block_pad.empty()) {
                    out.write(block_pad.data(), block_pad.size());
                }

                std::uint64_t gate_up_offset = out.tellp();
                out.write(gate_up_stage.data(), gate_up_bytes);

                std::uint64_t down_offset = out.tellp();
                out.write(down_stage.data(), down_bytes);

                index[l][e].gate_up_offset = gate_up_offset;
                index[l][e].gate_up_bytes = gate_up_bytes;
                index[l][e].down_offset = down_offset;
                index[l][e].down_bytes = down_bytes;
            }
        }

        // Rewrite final index and header
        out.seekp(0);
        write_header(out, header);
        for (int l = 0; l < moe_layers; ++l) {
            for (const SidecarExpertIndex& entry : index[l]) write_index(out, entry);
        }

        std::cout << "Successfully packed experts into: " << sidecar_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
