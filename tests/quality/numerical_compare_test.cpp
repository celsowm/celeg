
#include "support/numerical_compare.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using celeg::test::numerical::ComparisonResult;
using celeg::test::numerical::compare;
using celeg::test::numerical::top_k_agreement;
using celeg::test::numerical::require_within;

constexpr double kEps = 1e-12;

void run_identical_vectors_pass() {
    const std::vector<float> a = {0.1f, -0.2f, 0.5f, 0.8f, -0.4f};
    const std::vector<float> b = a;
    const ComparisonResult r = compare(a, b, /*top_k=*/2);
    if (std::fabs(r.cosine_similarity - 1.0) > kEps) {
        throw std::runtime_error("identical cosine != 1");
    }
    if (r.max_absolute_error > kEps) {
        throw std::runtime_error("identical max_abs != 0");
    }
    if (r.rmse > kEps) {
        throw std::runtime_error("identical rmse != 0");
    }
    if (std::fabs(r.top_k_agreement - 1.0) > kEps) {
        throw std::runtime_error("identical top_k_agreement != 1");
    }
    require_within(r, /*min_cosine=*/0.999, /*max_rmse=*/1e-9,
                   /*max_abs=*/1e-9, /*min_top_k_agreement=*/1.0, /*top_k=*/2);
}

void run_length_mismatch_rejected() {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f};
    const std::vector<float> b = {1.0f, 2.0f};
    bool threw = false;
    try { (void)compare(a, b); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) {
        throw std::runtime_error("compare did not reject length mismatch");
    }
}

void run_zero_vector_rejected() {
    const std::vector<float> a(4, 0.0f);
    const std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    bool threw = false;
    try { (void)compare(a, b); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) {
        throw std::runtime_error("compare did not reject zero-vector input");
    }
}

void run_quantization_tolerance_applied() {
    const std::vector<float> reference = {
        0.10f, -0.20f,  0.50f,  0.80f, -0.40f,
        0.30f,  0.60f, -0.10f,  0.20f,  0.70f,
    };
    const std::vector<float> quantized = {
        0.10f, -0.19f,  0.52f,  0.79f, -0.39f,
        0.31f,  0.59f, -0.10f,  0.21f,  0.69f,
    };
    const ComparisonResult r = compare(reference, quantized, /*top_k=*/3);
    require_within(r, /*min_cosine=*/0.999, /*max_rmse=*/0.05,
                   /*max_abs=*/0.25, /*min_top_k_agreement=*/1.0, /*top_k=*/3);
}

void run_tolerance_violation_reported() {
    const std::vector<float> a = {0.1f, -0.2f, 0.5f, 0.8f, -0.4f};
    std::vector<float> b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) b[i] = -a[i];
    const ComparisonResult r = compare(a, b);
    bool threw = false;
    try {
        require_within(r, /*min_cosine=*/0.999, /*max_rmse=*/1e-9,
                       /*max_abs=*/1e-9);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        throw std::runtime_error(
            "require_within did not fire on sign-flipped vector");
    }
}

void run_top_k_agreement_partial() {
    const std::vector<float> a = {0.1f, -0.2f, 0.5f, 0.8f, -0.4f, 0.6f};
    const std::vector<float> b = {0.1f, -0.2f, 0.45f, 0.81f, -0.4f, 0.61f};
    const double k3 = top_k_agreement(a, b, 3);
    if (k3 < 1.0 - kEps || k3 > 1.0 + kEps) {
        throw std::runtime_error(
            "top_k_agreement(3) should be 1.0 for the same top indices");
    }
    std::vector<float> c(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) c[i] = a[i];
    c[2] = -1.0f;
    c[4] = 0.7f;
    const double k3b = top_k_agreement(a, c, 3);
    if (std::fabs(k3b - (2.0 / 3.0)) > 1e-9) {
        throw std::runtime_error(
            "top_k_agreement partial mismatch: got " +
            std::to_string(k3b) + ", expected 0.6667");
    }
}

}

int main() {
    try {
        run_identical_vectors_pass();
        run_length_mismatch_rejected();
        run_zero_vector_rejected();
        run_quantization_tolerance_applied();
        run_tolerance_violation_reported();
        run_top_k_agreement_partial();
    } catch (const std::exception& e) {
        std::cerr << "numerical_compare_test: FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "numerical_compare_test: ok\n";
    return 0;
}
