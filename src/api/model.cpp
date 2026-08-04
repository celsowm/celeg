#include "api_internal.hpp"

#include <algorithm>
#include <vector>

extern "C" {

celeg_model* celeg_model_create(const char* path,
                                const celeg_cpu_model_options* options) {
    if (!path || !*path || !options) {
        celeg::api::global_error = "model path and options are required";
        return nullptr;
    }
    try {
        celeg::api::require_size(options->struct_size, sizeof(*options), "model options");
        celeg::api::require_size(options->generation.struct_size,
                                 sizeof(options->generation), "generation options");
        auto result = std::make_unique<celeg_model>();
        result->cpu = std::make_unique<celeg::CpuModel>(
            path, options->max_context, celeg::api::cpu_options(*options),
            celeg::api::generation(options->generation));
        return result.release();
    } catch (const std::exception& error) {
        celeg::api::global_error = error.what();
        return nullptr;
    }
}

void celeg_model_destroy(celeg_model* model) { delete model; }

celeg_status celeg_model_prefill(celeg_model* model,
                                 const int32_t* tokens, size_t count) {
    if (!tokens || count == 0) return CELEG_STATUS_INVALID_ARGUMENT;
    return celeg::api::protect(model, [&] {
        model->cpu->session().prefill(std::vector<int32_t>(tokens, tokens + count));
    });
}

celeg_status celeg_model_decode(celeg_model* model, int32_t* token) {
    if (!token) return CELEG_STATUS_INVALID_ARGUMENT;
    return celeg::api::protect(model, [&] {
        *token = model->cpu->session().decode();
    });
}

celeg_status celeg_model_copy_logits(celeg_model* model, float* output,
                                     size_t capacity, size_t* required) {
    if (!model || !required) return CELEG_STATUS_INVALID_ARGUMENT;
    return celeg::api::protect(model, [&] {
        const auto values = model->cpu->diagnostics().copy_logits();
        *required = values.size();
        if (!output || capacity < values.size()) {
            throw std::length_error("logit output buffer is too small");
        }
        std::copy(values.begin(), values.end(), output);
    });
}

celeg_status celeg_model_get_metrics(celeg_model* model, celeg_metrics* metrics) {
    if (!model || !metrics || metrics->struct_size < sizeof(*metrics)) {
        return CELEG_STATUS_INVALID_ARGUMENT;
    }
    return celeg::api::protect(model, [&] {
        const auto value = model->cpu->diagnostics().runtime_metrics();
        metrics->prefill_ms = value.last_prefill_ms;
        metrics->prefill_tokens = value.prefill_tokens;
        metrics->decode_ms = value.cumulative_decode_ms;
        metrics->decode_tokens = value.decoded_tokens;
    });
}

const char* celeg_model_last_error(const celeg_model* model) {
    return model ? model->error.c_str() : celeg::api::global_error.c_str();
}

} // extern "C"
