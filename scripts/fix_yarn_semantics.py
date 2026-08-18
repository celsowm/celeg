from pathlib import Path

ROOT = Path('.')

def replace(path: str, old: str, new: str, count: int = 1) -> None:
    file = ROOT / path
    text = file.read_text(encoding='utf-8')
    if old not in text:
        if new in text:
            return
        raise SystemExit(f'missing fragment in {path}: {old[:180]!r}')
    file.write_text(text.replace(old, new, count), encoding='utf-8')

replace('src/model/definition.cpp',
'''            if (!std::isfinite(value.beta_fast) || !std::isfinite(value.beta_slow) ||
                value.beta_fast < 0.0 || value.beta_slow < 0.0 ||
                value.beta_fast > value.beta_slow) {
                throw std::invalid_argument("invalid YaRN beta range");
            }''',
'''            if (value.original_context <= 0) {
                throw std::invalid_argument("YaRN requires an original context length");
            }
            if (!std::isfinite(value.beta_fast) || !std::isfinite(value.beta_slow) ||
                value.beta_fast <= 0.0 || value.beta_slow <= 0.0 ||
                value.beta_fast <= value.beta_slow) {
                throw std::invalid_argument("invalid YaRN beta range");
            }''')

replace('include/celeg/model/position.hpp',
'''            } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
                const double ramp = std::clamp(
                    (static_cast<double>(pair) - scaling.beta_fast) /
                        std::max(1.0, scaling.beta_slow - scaling.beta_fast),
                    0.0, 1.0);
                result /= 1.0 + ramp * (scaling.factor - 1.0);
                return result;''',
'''            } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
                constexpr double two_pi = 6.28318530717958647692;
                const auto correction_dimension = [&](double rotations) {
                    return static_cast<double>(rotary_dimension) *
                        std::log(static_cast<double>(scaling.original_context) /
                                 (rotations * two_pi)) /
                        (2.0 * std::log(base));
                };
                const double low = std::max(
                    0.0, std::floor(correction_dimension(scaling.beta_fast)));
                double high = std::min(
                    static_cast<double>(rotary_dimension / 2 - 1),
                    std::ceil(correction_dimension(scaling.beta_slow)));
                if (high <= low) high = low + 1.0e-3;
                const double ramp = std::clamp(
                    (static_cast<double>(pair) - low) / (high - low), 0.0, 1.0);
                const double extrapolation = 1.0 - ramp;
                const double interpolated = result / scaling.factor;
                result = interpolated * (1.0 - extrapolation) + result * extrapolation;
                return result;''')

replace('include/celeg/backend/cuda/kernels/rope.hpp',
'''        } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
            result.kind = 3;
            result.factor = static_cast<float>(scaling.factor);
            result.attention_factor = static_cast<float>(scaling.attention_factor);
            result.beta_fast = static_cast<float>(scaling.beta_fast);
            result.beta_slow = static_cast<float>(scaling.beta_slow);''',
'''        } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
            result.kind = 3;
            result.factor = static_cast<float>(scaling.factor);
            result.original_context = scaling.original_context;
            result.attention_factor = static_cast<float>(scaling.attention_factor);
            result.beta_fast = static_cast<float>(scaling.beta_fast);
            result.beta_slow = static_cast<float>(scaling.beta_slow);''')

replace('src/backend/cuda/kernels/rope.cuh',
'''    } else if (scaling.kind == 3) {
        const float span = fmaxf(1.0f, scaling.beta_slow - scaling.beta_fast);
        const float ramp = fminf(1.0f, fmaxf(0.0f,
            (static_cast<float>(pair) - scaling.beta_fast) / span));
        frequency /= 1.0f + ramp * (scaling.factor - 1.0f);''',
'''    } else if (scaling.kind == 3) {
        constexpr float two_pi = 6.28318530717958647692f;
        const float denominator = 2.0f * logf(theta);
        const float low_dimension = static_cast<float>(rotary_dimension) *
            logf(static_cast<float>(scaling.original_context) /
                 (scaling.beta_fast * two_pi)) / denominator;
        const float high_dimension = static_cast<float>(rotary_dimension) *
            logf(static_cast<float>(scaling.original_context) /
                 (scaling.beta_slow * two_pi)) / denominator;
        const float low = fmaxf(0.0f, floorf(low_dimension));
        float high = fminf(static_cast<float>(rotary_dimension / 2 - 1),
                           ceilf(high_dimension));
        if (high <= low) high = low + 1.0e-3f;
        const float ramp = fminf(1.0f, fmaxf(0.0f,
            (static_cast<float>(pair) - low) / (high - low)));
        const float extrapolation = 1.0f - ramp;
        const float interpolated = frequency / scaling.factor;
        frequency = interpolated * (1.0f - extrapolation) + frequency * extrapolation;''')

replace('src/model/inference/metadata.cpp',
'''                const double beta_fast = metadata.number_or("rope_scaling.beta_fast", 32.0);
                const double beta_slow = metadata.number_or("rope_scaling.beta_slow", 1.0);
                scaling = YarnRopeScaling{factor, attention_factor, beta_fast, beta_slow};''',
'''                const double beta_fast = metadata.number_or("rope_scaling.beta_fast", 32.0);
                const double beta_slow = metadata.number_or("rope_scaling.beta_slow", 1.0);
                const int original_context = static_cast<int>(metadata.integer_or(
                    "rope_scaling.original_max_position_embeddings", 0));
                if (original_context <= 0) {
                    inference_detail::fail(
                        ResolutionFailureKind::MissingRequiredMetadata,
                        "YaRN requires rope_scaling.original_max_position_embeddings");
                }
                scaling = YarnRopeScaling{
                    factor, original_context, attention_factor, beta_fast, beta_slow};''')

replace('src/model/descriptor/architecture.cpp',
'''            } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
                value.factor = scaling_number_value(checkpoint, "factor", 1.0);
                value.attention_factor = scaling_number_value(checkpoint, "attention_factor", 1.0);
                value.beta_fast = scaling_number_value(checkpoint, "beta_fast", 32.0);
                value.beta_slow = scaling_number_value(checkpoint, "beta_slow", 1.0);''',
'''            } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
                value.factor = scaling_number_value(checkpoint, "factor", 1.0);
                value.original_context = descriptor_.rope_scaling_original_context
                    ? descriptor_.rope_scaling_original_context(checkpoint)
                    : scaling_int_value(checkpoint, "original_max_position_embeddings", 0);
                value.attention_factor = scaling_number_value(checkpoint, "attention_factor", 1.0);
                value.beta_fast = scaling_number_value(checkpoint, "beta_fast", 32.0);
                value.beta_slow = scaling_number_value(checkpoint, "beta_slow", 1.0);''')

replace('src/model/resolved.cpp',
'''        } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
            out << "yarn:" << scaling.factor << ':' << scaling.attention_factor << ':'
                << scaling.beta_fast << ':' << scaling.beta_slow;''',
'''        } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
            out << "yarn:" << scaling.factor << ':' << scaling.original_context << ':'
                << scaling.attention_factor << ':' << scaling.beta_fast << ':'
                << scaling.beta_slow;''')

replace('tests/model_compiler_test.cpp',
'''    celeg::RopePositionSpec yarn_rope{10000.0, 1.0,
        celeg::YarnRopeScaling{2.0, 1.25, 1.0, 4.0}};''',
'''    celeg::RopePositionSpec yarn_rope{10000.0, 1.0,
        celeg::YarnRopeScaling{2.0, 32, 1.25, 32.0, 1.0}};''')

replace('tests/automatic_inference_test.cpp',
'''    yarn_metadata.values["rope_scaling.factor"] = 4.0;''',
'''    yarn_metadata.values["rope_scaling.factor"] = 4.0;
    yarn_metadata.values["rope_scaling.original_max_position_embeddings"] = int64_t(8192);''')
replace('tests/automatic_inference_test.cpp',
'''    CELEG_TEST_CHECK(std::holds_alternative<celeg::YarnRopeScaling>(yarn_rope->scaling));''',
'''    CELEG_TEST_CHECK(std::holds_alternative<celeg::YarnRopeScaling>(yarn_rope->scaling));
    CELEG_TEST_CHECK(std::get<celeg::YarnRopeScaling>(yarn_rope->scaling).original_context == 8192);''')

print('YaRN canonical semantics now include original context and correction range')