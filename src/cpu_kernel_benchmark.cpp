#include "lfm/cpu_isa.hpp"
#include "lfm/cpu_kernels.hpp"
#include "lfm/cpu_quantization.hpp"
#include "lfm/cpu_thread_pool.hpp"
#include "lfm/cpu_topology.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        const size_t rows = argc > 1 ? static_cast<size_t>(std::stoull(argv[1])) : 4096;
        const size_t cols = argc > 2 ? static_cast<size_t>(std::stoull(argv[2])) : 1024;
        const int iterations = argc > 3 ? std::stoi(argv[3]) : 10;
        const size_t threads = argc > 4 ? static_cast<size_t>(std::stoull(argv[4])) : 0;
        const int group = argc > 5 ? std::stoi(argv[5]) : 32;
        const size_t batch = argc > 6 ? static_cast<size_t>(std::stoull(argv[6])) : 1;
        const lfm::CpuIsa requested = argc > 7
            ? lfm::parse_cpu_isa(argv[7]) : lfm::CpuIsa::Auto;
        const lfm::CpuAffinityPolicy affinity = argc > 8
            ? lfm::parse_cpu_affinity(argv[8]) : lfm::CpuAffinityPolicy::None;
        if (rows == 0 || cols == 0 || batch == 0 || iterations <= 0 ||
            (group != 32 && group != 64)) {
            throw std::invalid_argument("invalid benchmark arguments");
        }
        std::vector<float> source(rows * cols);
        std::vector<float> input(batch * cols);
        std::vector<float> output(batch * rows);
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = std::sin(static_cast<float>(i) * 0.001f);
        }
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::cos(static_cast<float>(i) * 0.01f);
        }
        const auto matrix = lfm::quantize_float_groupwise_q4(
            source.data(), rows, cols, static_cast<size_t>(group));
        const auto caps = lfm::detect_cpu_capabilities();
        const lfm::CpuIsa isa = requested == lfm::CpuIsa::Auto
            ? caps.best_isa() : requested;
        if (!lfm::cpu_isa_compiled(isa)) {
            throw std::invalid_argument("requested ISA was not compiled into this binary");
        }
        if (!caps.supports(isa)) {
            throw std::invalid_argument("requested ISA is not supported by this host");
        }
        if (isa == lfm::CpuIsa::AmxInt8 || isa == lfm::CpuIsa::DotProd ||
            isa == lfm::CpuIsa::I8mm || isa == lfm::CpuIsa::Sve2 ||
            isa == lfm::CpuIsa::Sme2) {
            throw std::invalid_argument("requested ISA has no executable v0.0.20 kernel");
        }
        lfm::CpuThreadPool pool(threads, affinity);
        lfm::CpuLinearEngine linear(isa, pool);
        auto execute = [&] {
            if (batch == 1) linear.gemv(matrix, input.data(), output.data());
            else linear.gemm(matrix, input.data(), output.data(), batch);
        };
        for (int i = 0; i < 2; ++i) execute();
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) execute();
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
        const double calls_per_second = iterations * 1000.0 / ms;
        const double rows_per_second = calls_per_second * batch;
        const double weight_gb = static_cast<double>(matrix.memory_bytes()) / 1e9;
        std::cout << std::fixed << std::setprecision(3)
                  << "cpu.isa=" << lfm::cpu_isa_name(isa) << '\n'
                  << "cpu.threads=" << (pool.size() + 1) << '\n'
                  << "cpu.affinity=" << lfm::cpu_affinity_name(affinity) << '\n'
                  << "cpu.pinned_workers=" << pool.pinned_workers() << '\n'
                  << "matrix.rows=" << rows << '\n'
                  << "matrix.cols=" << cols << '\n'
                  << "matrix.q4_group=" << group << '\n'
                  << "matrix.batch=" << batch << '\n'
                  << "benchmark.iterations=" << iterations << '\n'
                  << "benchmark.elapsed_ms=" << ms << '\n'
                  << "benchmark.calls_per_second=" << calls_per_second << '\n'
                  << "benchmark.batch_rows_per_second=" << rows_per_second << '\n'
                  << "benchmark.effective_weight_gb_per_second="
                  << weight_gb * rows_per_second << '\n'
                  << "checksum=" << output[(batch / 2) * rows + rows / 2] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
