#include "lfm/cpu_model.hpp"
#include "lfm/cpu_isa.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) {
            std::cerr << "usage: lfm25-cpu-pack MODEL_SAFETENSORS [CACHE_DIR] [GROUP=32|64] [ISA=auto|scalar|avx2|neon]\n";
            return 2;
        }
        lfm::CpuModelOptions options;
        if (argc >= 3 && std::string(argv[2]) != "-") options.pack_cache_directory = argv[2];
        if (argc >= 4) {
            const int group = std::stoi(argv[3]);
            if (group != 32 && group != 64) throw std::invalid_argument("group must be 32 or 64");
            options.weight_format = group == 64 ? lfm::CpuWeightFormat::Q4Group64
                                                : lfm::CpuWeightFormat::Q4Group32;
        }
        if (argc >= 5) options.isa = lfm::parse_cpu_isa(argv[4]);
        options.use_pack_cache = true;
        options.threads = 1;
        lfm::GenerationConfig generation;
        lfm::CpuModel model(argv[1], 1, options, generation);
        std::cout << "backend=" << model.diagnostics().backend_description() << '\n'
                  << "pack_path=" << model.diagnostics().pack_path() << '\n'
                  << "cache_status=" << (model.diagnostics().loaded_from_pack() ? "hit" : "created") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
