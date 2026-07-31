// Native throughput benchmark for the CPU engine, built to be directly
// comparable to llama.cpp's llama-bench: same synthetic-token methodology
// (random/dummy token ids, not a real prompt) and the same JSON row schema
// (n_prompt, n_gen, avg_ns, stddev_ns, avg_ts, stddev_ts) so a single
// benchmarks/compare.py loader can parse either engine's output. Running it
// in-process (load once, reset the session between reps) rather than
// shelling out to celeg-cpu-run per rep avoids paying process-startup and
// model-load cost inside the timed region, which is what llama-bench does
// too.
#include "celeg/backend/cpu/isa.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/topology.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model;
    std::string isa = "auto";
    int n_prompt = 512;
    int n_gen = 128;
    int reps = 5;
    int warmup = 1;
    int threads = 0;
    int batch_size = 256;
    int ubatch_size = 256;
    int group_size = 32;
    bool group_size_explicit = false;
    int context = 0;
    uint64_t seed = 1;
    std::string output = "json";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + key);
            return argv[i];
        };
        if (key == "--model" || key == "-m") args.model = value();
        else if (key == "--n-prompt" || key == "-p") args.n_prompt = std::stoi(value());
        else if (key == "--n-gen" || key == "-n") args.n_gen = std::stoi(value());
        else if (key == "--reps" || key == "-r") args.reps = std::stoi(value());
        else if (key == "--warmup") args.warmup = std::stoi(value());
        else if (key == "--threads" || key == "-t") args.threads = std::stoi(value());
        else if (key == "--batch-size" || key == "-b") args.batch_size = std::stoi(value());
        else if (key == "--ubatch-size" || key == "-ub") args.ubatch_size = std::stoi(value());
        else if (key == "--cpu-q4-group") {
            args.group_size = std::stoi(value());
            args.group_size_explicit = true;
        }
        else if (key == "--cpu-isa") args.isa = value();
        else if (key == "--context") args.context = std::stoi(value());
        else if (key == "--seed") args.seed = std::stoull(value());
        else if (key == "--output" || key == "-o") args.output = value();
        else if (key == "--help") {
            std::cout
                << "celeg-bench --model DIR [options]\n"
                << "  -p, --n-prompt N     synthetic prefill tokens (default 512)\n"
                << "  -n, --n-gen N        decode steps (default 128)\n"
                << "  -r, --reps N         timed repetitions (default 5)\n"
                << "  --warmup N           untimed warmup repetitions (default 1)\n"
                << "  -t, --threads N      thread count (default 0 = auto)\n"
                << "  -b, --batch-size N   prefill chunk size (default 256)\n"
                << "  -ub, --ubatch-size N physical chunk size (must equal batch)\n"
                << "  --cpu-q4-group 32|64 weight quantization group size\n"
                << "  --cpu-isa auto|scalar|avx2|avx-vnni|avx512-vnni|neon\n"
                << "  --context N          override context size\n"
                << "  -o, --output json|md output format (default json)\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + key);
    }
    if (args.model.empty()) throw std::runtime_error("--model is required");
    if (args.n_prompt < 0 || args.n_gen < 0 || args.reps <= 0 || args.warmup < 0 ||
        args.threads < 0 || args.batch_size <= 0 || args.ubatch_size <= 0 ||
        args.batch_size != args.ubatch_size ||
        (args.group_size != 32 && args.group_size != 64)) {
        throw std::runtime_error("invalid numeric argument");
    }
    if (args.context <= 0) args.context = args.n_prompt + args.n_gen + 16;
    if (args.group_size_explicit &&
        std::filesystem::path(args.model).extension() == ".gguf") {
        throw std::runtime_error(
            "--cpu-q4-group is only valid for Safetensors checkpoints");
    }
    return args;
}

std::vector<int32_t> synthetic_tokens(int vocab_size, int count) {
    std::vector<int32_t> tokens(count);
    for (auto& token : tokens) token = std::rand() % vocab_size;
    return tokens;
}

struct Sample {
    double ns = 0.0;
    double tokens_per_second = 0.0;
};

struct Stats {
    int n_tokens = 0;
    double avg_ns = 0.0;
    double stddev_ns = 0.0;
    double avg_ts = 0.0;
    double stddev_ts = 0.0;
};

Stats summarize(const std::vector<Sample>& samples, int n_tokens) {
    Stats stats;
    stats.n_tokens = n_tokens;
    const size_t count = samples.size();
    double sum_ns = 0.0, sum_ts = 0.0;
    for (const auto& s : samples) { sum_ns += s.ns; sum_ts += s.tokens_per_second; }
    stats.avg_ns = sum_ns / static_cast<double>(count);
    stats.avg_ts = sum_ts / static_cast<double>(count);
    if (count > 1) {
        double var_ns = 0.0, var_ts = 0.0;
        for (const auto& s : samples) {
            var_ns += (s.ns - stats.avg_ns) * (s.ns - stats.avg_ns);
            var_ts += (s.tokens_per_second - stats.avg_ts) * (s.tokens_per_second - stats.avg_ts);
        }
        stats.stddev_ns = std::sqrt(var_ns / static_cast<double>(count - 1));
        stats.stddev_ts = std::sqrt(var_ts / static_cast<double>(count - 1));
    }
    return stats;
}

void print_json_row(std::ostream& out, bool first, const std::string& model_filename,
                     const std::string& cpu_info, int n_prompt, int n_gen, const Stats& stats) {
    if (!first) out << ",\n";
    out << "  {\n"
        << "    \"model_filename\": \"" << model_filename << "\",\n"
        << "    \"cpu_info\": \"" << cpu_info << "\",\n"
        << "    \"n_prompt\": " << n_prompt << ",\n"
        << "    \"n_gen\": " << n_gen << ",\n"
        << "    \"avg_ns\": " << std::fixed << std::setprecision(0) << stats.avg_ns << ",\n"
        << "    \"stddev_ns\": " << stats.stddev_ns << ",\n"
        << "    \"avg_ts\": " << std::setprecision(6) << stats.avg_ts << ",\n"
        << "    \"stddev_ts\": " << stats.stddev_ts << "\n"
        << "  }";
}

std::string escape_json(const std::string& text) {
    std::string escaped;
    for (char c : text) {
        if (c == '"' || c == '\\') escaped += '\\';
        escaped += c;
    }
    return escaped;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const celeg::CpuCapabilities caps = celeg::detect_cpu_capabilities();
        const std::string cpu_info = escape_json(caps.summary());

        celeg::CpuModelOptions options;
        options.isa = celeg::parse_cpu_isa(args.isa);
        options.weight_format = args.group_size == 64
            ? celeg::CpuWeightFormat::Q4Group64 : celeg::CpuWeightFormat::Q4Group32;
        options.threads = static_cast<size_t>(args.threads);
        options.prefill_chunk_tokens = static_cast<size_t>(args.ubatch_size);
        celeg::GenerationConfig generation;
        generation.temperature = 0.0f;
        generation.top_k = 1;

        celeg::CpuModel engine(args.model, args.context, options, generation);
        const int vocab_size = engine.diagnostics().vocab_size();
        std::srand(static_cast<unsigned int>(args.seed));

        std::vector<Sample> prompt_samples;
        std::vector<Sample> gen_samples;

        if (args.n_prompt > 0) {
            const auto tokens = synthetic_tokens(vocab_size, args.n_prompt);
            for (int i = 0; i < args.warmup; ++i) {
                engine.session().reset();
                engine.session().prefill(tokens);
            }
            for (int i = 0; i < args.reps; ++i) {
                engine.session().reset();
                engine.diagnostics().clear_runtime_metrics();
                engine.session().prefill(tokens);
                const auto metrics = engine.diagnostics().runtime_metrics();
                prompt_samples.push_back({metrics.last_prefill_ms * 1e6,
                                           metrics.prefill_tokens_per_second()});
            }
        }

        if (args.n_gen > 0) {
            const auto gen_tokens = synthetic_tokens(vocab_size, args.n_gen + 1);
            const std::vector<int32_t> prime = {gen_tokens[0]};
            auto run_gen = [&]() {
                engine.session().reset();
                engine.session().prefill(prime);
                for (int j = 0; j < args.n_gen; ++j) {
                    engine.session().eval_token(gen_tokens[static_cast<size_t>(j) + 1]);
                }
            };
            for (int i = 0; i < args.warmup; ++i) run_gen();
            for (int i = 0; i < args.reps; ++i) {
                engine.session().reset();
                engine.diagnostics().clear_runtime_metrics();
                engine.session().prefill(prime);
                for (int j = 0; j < args.n_gen; ++j) {
                    engine.session().eval_token(gen_tokens[static_cast<size_t>(j) + 1]);
                }
                const auto metrics = engine.diagnostics().runtime_metrics();
                gen_samples.push_back({metrics.cumulative_decode_ms * 1e6,
                                        metrics.decode_tokens_per_second()});
            }
        }

        if (std::getenv("CELEG_PRINT_PROFILE") && args.n_prompt > 0) {
            const auto profile = engine.diagnostics().prefill_profile();
            std::cerr << "profile.linear_ms=" << profile.linear_ms << '\n'
                      << "profile.attention_ms=" << profile.attention_ms << '\n'
                      << "profile.shortconv_ms=" << profile.shortconv_ms << '\n'
                      << "profile.moe_router_ms=" << profile.moe_router_ms << '\n'
                      << "profile.moe_expert_ms=" << profile.moe_expert_ms << '\n'
                      << "profile.total_ms=" << profile.total_ms << '\n';
        }

        const std::string model_filename = escape_json(args.model);
        if (args.output == "json") {
            std::cout << "[\n";
            bool first = true;
            if (!prompt_samples.empty()) {
                print_json_row(std::cout, first, model_filename, cpu_info,
                               args.n_prompt, 0, summarize(prompt_samples, args.n_prompt));
                first = false;
            }
            if (!gen_samples.empty()) {
                print_json_row(std::cout, first, model_filename, cpu_info,
                               0, args.n_gen, summarize(gen_samples, args.n_gen));
                first = false;
            }
            std::cout << "\n]\n";
        } else {
            std::cout << "| test | tok/s |\n|---|---|\n";
            if (!prompt_samples.empty()) {
                const auto stats = summarize(prompt_samples, args.n_prompt);
                std::cout << "| pp" << args.n_prompt << " | "
                          << std::fixed << std::setprecision(2) << stats.avg_ts
                          << " +/- " << stats.stddev_ts << " |\n";
            }
            if (!gen_samples.empty()) {
                const auto stats = summarize(gen_samples, args.n_gen);
                std::cout << "| tg" << args.n_gen << " | "
                          << std::fixed << std::setprecision(2) << stats.avg_ts
                          << " +/- " << stats.stddev_ts << " |\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
