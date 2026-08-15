#include "celeg/checkpoint/repositories/safetensors.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/detail/binary_codec.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"

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
    celeg::binary::write_le(output, header.num_layers);
    celeg::binary::write_le(output, header.num_experts);
    celeg::binary::write_le(output, header.moe_intermediate);
    celeg::binary::write_le(output, header.hidden);
    for (const std::uint64_t value : header.reserved) {
        celeg::binary::write_le(output, value);
    }
}

void write_index(std::ostream& output, const SidecarExpertIndex& index) {
    celeg::binary::write_le(output, index.gate_up_offset);
    celeg::binary::write_le(output, index.gate_up_bytes);
    celeg::binary::write_le(output, index.down_offset);
    celeg::binary::write_le(output, index.down_bytes);
}

std::string layer_name(int layer, const std::string& suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: celeg-pack-experts SAFETENSORS_PATH OUTPUT_SIDECAR_PATH\n";
        return 2;
    }

    try {
        std::filesystem::path model_path(argv[1]);
        std::filesystem::path sidecar_path(argv[2]);

        const celeg::detail::ModelBootstrap bootstrap =
            celeg::detail::load_model_bootstrap(model_path);
        const auto& model = bootstrap.model;

        std::vector<int> moe_layer_ids;
        for (int layer = 0; layer < static_cast<int>(model.graph.layers.size()); ++layer) {
            if (std::holds_alternative<celeg::MixtureOfExpertsSpec>(
                    model.graph.layers.at(static_cast<size_t>(layer)).feed_forward)) {
                moe_layer_ids.push_back(layer);
            }
        }
        const int moe_layers = static_cast<int>(moe_layer_ids.size());
        if (moe_layers == 0) {
            std::cout << "No MoE layers in this model, nothing to pack.\n";
            return 0;
        }
        const auto& first_moe = std::get<celeg::MixtureOfExpertsSpec>(
            model.graph.layers.at(static_cast<size_t>(moe_layer_ids.front())).feed_forward);
        const int expert_count = first_moe.num_experts;
        const int moe_intermediate = first_moe.intermediate_size;

        std::cout << "Packing MoE experts for " << model.provenance.identity << "...\n"
                  << "  MoE layers: " << moe_layers << "\n"
                  << "  Experts/layer: " << expert_count << "\n"
                  << "  Intermediate: " << moe_intermediate << "\n"
                  << "  Hidden: " << model.graph.hidden << "\n";

        celeg::SafeTensorRepository repo(model_path);

        SidecarHeader header;
        header.num_layers = static_cast<std::uint32_t>(moe_layers);
        header.num_experts = static_cast<std::uint32_t>(expert_count);
        header.moe_intermediate = static_cast<std::uint64_t>(moe_intermediate);
        header.hidden = static_cast<std::uint64_t>(model.graph.hidden);

        std::vector<std::vector<SidecarExpertIndex>> index(
            moe_layers, std::vector<SidecarExpertIndex>(expert_count));

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

        const size_t moe_inter = static_cast<size_t>(moe_intermediate);
        const size_t hidden_c = static_cast<size_t>(model.graph.hidden);
        const size_t gate_up_elems = 2 * moe_inter * hidden_c;
        const size_t down_elems = hidden_c * moe_inter;
        const size_t gate_up_bytes = gate_up_elems * 2; // BF16 element size = 2
        const size_t down_bytes = down_elems * 2;
        const size_t w_bytes = moe_inter * hidden_c * 2;

        std::vector<char> gate_up_stage(gate_up_bytes);
        std::vector<char> down_stage(down_bytes);

        for (int l = 0; l < moe_layers; ++l) {
            const int actual_layer_idx = moe_layer_ids[static_cast<size_t>(l)];
            std::cout << "Packing layer " << actual_layer_idx << " (" << (l + 1) << "/" << moe_layers << ")...\n";

            for (int e = 0; e < expert_count; ++e) {
                const std::string w1_name = layer_name(
                    actual_layer_idx, "feed_forward.experts." + std::to_string(e) + ".w1.weight");
                const std::string w3_name = layer_name(
                    actual_layer_idx, "feed_forward.experts." + std::to_string(e) + ".w3.weight");
                const std::string w2_name = layer_name(
                    actual_layer_idx, "feed_forward.experts." + std::to_string(e) + ".w2.weight");

                celeg::HostTensorView w1 = repo.tensor(w1_name);
                celeg::HostTensorView w3 = repo.tensor(w3_name);
                celeg::HostTensorView w2 = repo.tensor(w2_name);

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
