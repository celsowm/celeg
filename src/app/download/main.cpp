#include "celeg/checkpoint/downloader.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        celeg::DownloadOptions opts;
        opts.repo_id = "LiquidAI/LFM2.5-230M";
        opts.revision = "main";

        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("missing value for " + key);
                return argv[i];
            };
            if (key == "--revision" || key == "-r") opts.revision = value();
            else if (key == "--cache-dir") opts.cache_dir = value();
            else if (key == "--file") opts.files.push_back(value());
            else if (key == "--force") opts.force = true;
            else if (key == "--quiet" || key == "-q") opts.quiet = true;
            else if (key == "--help" || key == "-h") {
                std::cout
                    << "celeg-download [REPO_ID] [options]\n"
                    << "  REPO_ID defaults to LiquidAI/LFM2.5-230M\n"
                    << "  Common checkpoints:\n"
                    << "    LiquidAI/LFM2.5-230M            (230M base)\n"
                    << "    LiquidAI/LFM2.5-1.2B-Instruct   (1.2B Instruct)\n"
                    << "  --revision REV    (default: main)\n"
                    << "  --cache-dir DIR   (default: " << celeg::default_hf_cache_dir().string() << ")\n"
                    << "  --file FILE       (repeatable; default: all files)\n"
                    << "  --force           re-download even if cached\n"
                    << "  --quiet           suppress progress output\n";
                return 0;
            } else if (key.size() > 0 && key[0] == '-') {
                throw std::runtime_error("unknown option: " + key);
            } else {
                opts.repo_id = key;
            }
        }

        celeg::DownloadResult result = celeg::download_model(opts);
        if (!opts.quiet) {
            std::cerr << "done: " << result.snapshot_path.string() << "\n";
        }
        std::cout << result.snapshot_path.string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
