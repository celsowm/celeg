#include "celeg/model/position.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

bool near(double actual, double expected, double relative = 1.0e-12) {
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    return std::abs(actual - expected) <= relative * scale;
}

double base_frequency(double theta, int pair, int dimension) {
    return std::pow(theta, -2.0 * static_cast<double>(pair) /
                               static_cast<double>(dimension));
}

void check_no_scaling() {
    celeg::RopePositionSpec spec;
    spec.theta = 10000.0;
    spec.scaling = celeg::NoRopeScaling{};
    for (const int pair : {0, 1, 3}) {
        assert(near(celeg::rope_frequency(spec, pair, 8, 4096),
                    base_frequency(spec.theta, pair, 8)));
    }
}

void check_linear() {
    celeg::RopePositionSpec spec;
    spec.theta = 10000.0;
    spec.scaling = celeg::LinearRopeScaling{4.0};
    for (const int pair : {0, 1, 3}) {
        assert(near(celeg::rope_frequency(spec, pair, 8, 4096),
                    base_frequency(spec.theta, pair, 8) / 4.0));
    }
}

void check_dynamic_ntk() {
    celeg::RopePositionSpec spec;
    spec.theta = 10000.0;
    spec.scaling = celeg::DynamicNtkRopeScaling{4.0, 128};

    for (const int position : {0, 127, 128}) {
        for (const int pair : {0, 1, 3}) {
            assert(near(celeg::rope_frequency(spec, pair, 8, position),
                        base_frequency(spec.theta, pair, 8)));
        }
    }

    constexpr int position = 256;
    const double ratio = 4.0 * static_cast<double>(position) / 128.0 - 3.0;
    const double adjusted_theta = spec.theta * std::pow(ratio, 8.0 / 6.0);
    for (const int pair : {0, 1, 2, 3}) {
        const double expected = base_frequency(adjusted_theta, pair, 8);
        const double actual = celeg::rope_frequency(spec, pair, 8, position);
        assert(near(actual, expected));
    }

    // Pair zero is invariant under a theta/base change. The former host
    // implementation multiplied the completed frequency and therefore made
    // pair zero greater than one after the context boundary.
    assert(near(celeg::rope_frequency(spec, 0, 8, position), 1.0));
}

void check_yarn() {
    celeg::RopePositionSpec spec;
    spec.theta = 10000.0;
    spec.scaling = celeg::YarnRopeScaling{4.0, 1.3, 32.0, 1.0, 4096};

    const auto correction_dimension = [&](double rotations) {
        return 8.0 * std::log(4096.0 / (rotations * 2.0 * kPi)) /
            (2.0 * std::log(spec.theta));
    };
    double low = std::max(0.0, std::floor(correction_dimension(32.0)));
    double high = std::min(7.0, std::ceil(correction_dimension(1.0)));
    if (low == high) high += 0.001;
    for (const int pair : {0, 1, 3}) {
        const double base = base_frequency(spec.theta, pair, 8);
        const double ramp = std::clamp((static_cast<double>(pair) - low) /
                                           (high - low),
                                       0.0, 1.0);
        const double extrapolation = 1.0 - ramp;
        const double expected = base *
            (extrapolation + (1.0 - extrapolation) / 4.0);
        assert(near(celeg::rope_frequency(spec, pair, 8, 8192), expected));
    }
    assert(std::abs(celeg::rope_attention_scale(spec, 8192) - 1.69f) < 1.0e-6f);
}

void check_longrope() {
    celeg::RopePositionSpec spec;
    spec.theta = 10000.0;
    celeg::LongRopeScaling scaling;
    scaling.original_context = 128;
    scaling.short_factors = {1.0f, 2.0f, 3.0f, 4.0f};
    scaling.long_factors = {2.0f, 4.0f, 6.0f, 8.0f};
    spec.scaling = scaling;

    for (const int pair : {0, 1, 3}) {
        const double base = base_frequency(spec.theta, pair, 8);
        assert(near(celeg::rope_frequency(spec, pair, 8, 128),
                    base / scaling.short_factors[static_cast<size_t>(pair)]));
        assert(near(celeg::rope_frequency(spec, pair, 8, 129),
                    base / scaling.long_factors[static_cast<size_t>(pair)]));
    }
}

void check_llama3() {
    celeg::RopePositionSpec spec;
    spec.theta = 10000.0;
    celeg::Llama3FrequencyScaling scaling;
    scaling.factor = 8.0;
    scaling.original_context = 8192;
    scaling.low_frequency_factor = 1.0;
    scaling.high_frequency_factor = 4.0;
    spec.scaling = scaling;

    for (const int pair : {0, 1, 2, 3}) {
        double expected = base_frequency(spec.theta, pair, 8);
        const double wavelength = 2.0 * kPi / expected;
        if (wavelength > 8192.0) {
            expected /= 8.0;
        } else if (wavelength >= 2048.0) {
            const double blend = std::clamp(
                (wavelength * 4.0 / 8192.0 - 1.0) / 3.0, 0.0, 1.0);
            expected /= 1.0 + blend * 7.0;
        }
        assert(near(celeg::rope_frequency(spec, pair, 8, 4096), expected));
    }
}

void check_proportional() {
    celeg::RopePositionSpec spec;
    spec.theta = 10000.0;
    spec.rotary_fraction = 0.5;
    spec.scaling = celeg::ProportionalRopeScaling{2.0};
    constexpr int rotary_dimension = 4;
    for (const int pair : {0, 1}) {
        const double base = base_frequency(spec.theta, pair, rotary_dimension);
        const double expected = std::pow(base, 0.5) / 2.0;
        assert(near(celeg::rope_frequency(
            spec, pair, rotary_dimension, 2048), expected));
    }
}

}

int main() {
    check_no_scaling();
    check_linear();
    check_dynamic_ntk();
    check_yarn();
    check_longrope();
    check_llama3();
    check_proportional();
    return 0;
}
