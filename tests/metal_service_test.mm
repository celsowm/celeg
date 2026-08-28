#include "celeg/backend/metal/runtime_types.hpp"
#include "celeg/serve/metal_inference_service.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

celeg::serve::GenerateRequest request() {
    celeg::serve::GenerateRequest result;
    result.prompt_tokens = {1, 36309};
    result.max_output_tokens = 2;
    result.eos_token_ids = {-1};
    result.generation.temperature = 0.0f;
    result.generation.top_k = 1;
    return result;
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: metal_service_test MODEL");
        celeg::MetalEngineOptions options;
        options.max_active_requests = 2;
        options.max_batched_tokens = 4;
        options.prefill_chunk_tokens = 2;
        celeg::serve::MetalInferenceService service(argv[1], 128, {}, options);
        const celeg::RequestId first = service.submit(request());
        const celeg::RequestId second = service.submit(request());
        service.start();
        std::vector<int32_t> first_tokens;
        std::vector<int32_t> second_tokens;
        for (int step = 0; step < 16; ++step) {
            service.step();
            const auto first_event = service.poll(first, 0);
            const auto second_event = service.poll(second, 0);
            first_tokens.insert(first_tokens.end(), first_event.tokens.begin(),
                                first_event.tokens.end());
            second_tokens.insert(second_tokens.end(), second_event.tokens.begin(),
                                 second_event.tokens.end());
            if (first_event.finished && second_event.finished) break;
        }
        service.stop();
        if (first_tokens.size() != 2 || second_tokens.size() != 2 ||
            service.metrics().completed_requests != 2) {
            throw std::runtime_error("Metal concurrent request scheduling failed");
        }
        if (!service.release(first) || !service.release(second)) {
            throw std::runtime_error("Metal request release failed");
        }
        std::cout << "metal concurrent requests generated 2 tokens each\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
