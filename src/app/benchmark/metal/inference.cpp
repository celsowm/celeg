#include "celeg/backend/metal/model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string model;
    int context = 128;
    int prompt_tokens = 32;
    int decode_tokens = 8;
    int warmup = 1;
    int repetitions = 3;
};

std::string value(int& index, int argc, char** argv, const std::string& key) {
    if (++index >= argc) throw std::invalid_argument("missing value for " + key);
    return argv[index];
}

Arguments parse(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--model") result.model = value(index, argc, argv, key);
        else if (key == "--context") result.context = std::stoi(value(index, argc, argv, key));
        else if (key == "--prompt-tokens") result.prompt_tokens = std::stoi(value(index, argc, argv, key));
        else if (key == "--decode-tokens") result.decode_tokens = std::stoi(value(index, argc, argv, key));
        else if (key == "--warmup") result.warmup = std::stoi(value(index, argc, argv, key));
        else if (key == "--repetitions") result.repetitions = std::stoi(value(index, argc, argv, key));
        else if (key == "--help") {
            std::cout << "celeg-metal-bench --model PATH [--context N] "
                         "[--prompt-tokens N] [--decode-tokens N] "
                         "[--warmup N] [--repetitions N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (result.model.empty()) throw std::invalid_argument("--model is required");
    if (result.context <= 0 || result.prompt_tokens <= 0 || result.decode_tokens < 0 ||
        result.warmup < 0 || result.repetitions <= 0 ||
        result.prompt_tokens + result.decode_tokens > result.context) {
        throw std::invalid_argument("invalid benchmark dimensions");
    }
    return result;
}

struct Sample {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
};

Sample run(celeg::MetalModel& model,
           const std::vector<int32_t>& prompt,
           const std::vector<int32_t>& decode) {
    auto session = model.session();
    const auto prefill_start = std::chrono::steady_clock::now();
    session.prefill(prompt);
    const auto prefill_end = std::chrono::steady_clock::now();
    const auto decode_start = std::chrono::steady_clock::now();
    for (const int32_t token : decode) session.eval_token(token);
    const auto decode_end = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count(),
        std::chrono::duration<double, std::milli>(decode_end - decode_start).count()};
}

double average(const std::vector<Sample>& samples, double Sample::*field) {
    double total = 0.0;
    for (const Sample& sample : samples) total += sample.*field;
    return total / static_cast<double>(samples.size());
}

std::string json_string(const std::string& value) {
    std::ostringstream output;
    output << std::quoted(value);
    return output.str();
}

}

int main(int argc, char** argv) {
    try {
        const Arguments args = parse(argc, argv);
        celeg::GenerationConfig generation;
        generation.temperature = 0.0f;
        generation.top_k = 1;
        generation.top_p = 1.0f;
        generation.repetition_penalty = 1.0f;
        celeg::MetalModel model(args.model, args.context, {}, generation);
        const int vocab = model.vocab_size();
        std::vector<int32_t> prompt(static_cast<size_t>(args.prompt_tokens));
        std::vector<int32_t> decode(static_cast<size_t>(args.decode_tokens));
        for (int index = 0; index < args.prompt_tokens; ++index) {
            prompt[static_cast<size_t>(index)] = index % vocab;
        }
        for (int index = 0; index < args.decode_tokens; ++index) {
            decode[static_cast<size_t>(index)] = (args.prompt_tokens + index) % vocab;
        }
        for (int index = 0; index < args.warmup; ++index) {
            static_cast<void>(run(model, prompt, decode));
        }
        std::vector<Sample> samples;
        samples.reserve(static_cast<size_t>(args.repetitions));
        for (int index = 0; index < args.repetitions; ++index) {
            samples.push_back(run(model, prompt, decode));
        }
        const double prefill_ms = average(samples, &Sample::prefill_ms);
        const double decode_ms = average(samples, &Sample::decode_ms);
        const double prefill_rate = static_cast<double>(args.prompt_tokens) * 1000.0 / prefill_ms;
        const double decode_rate = args.decode_tokens == 0
            ? 0.0 : static_cast<double>(args.decode_tokens) * 1000.0 / decode_ms;
        std::cout << "{\n"
                  << "  \"model\": " << json_string(std::filesystem::path(args.model).string()) << ",\n"
                  << "  \"backend\": " << json_string(model.backend_description()) << ",\n"
                  << "  \"context\": " << args.context << ",\n"
                  << "  \"prompt_tokens\": " << args.prompt_tokens << ",\n"
                  << "  \"decode_tokens\": " << args.decode_tokens << ",\n"
                  << "  \"warmup\": " << args.warmup << ",\n"
                  << "  \"repetitions\": " << args.repetitions << ",\n"
                  << "  \"prefill_ms\": " << std::setprecision(10) << prefill_ms << ",\n"
                  << "  \"prefill_tokens_per_second\": " << prefill_rate << ",\n"
                  << "  \"decode_ms\": " << decode_ms << ",\n"
                  << "  \"decode_tokens_per_second\": " << decode_rate << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
