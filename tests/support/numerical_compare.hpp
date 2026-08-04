#pragma once

// Numerical comparison utilities used by Phase 0 baseline tests and, later, by
// the Phase 15 testing matrix. Pure host code, header-only, no CUDA dependency.
// See docs/ARCHITECTURE_EVIDENCE.md (tolerance classes) and
// section 15.3 (quantization quality).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace celeg::test::numerical {

struct ComparisonResult {
    double cosine_similarity = 0.0;
    double max_absolute_error = 0.0;
    double rmse = 0.0;
    double top_k_agreement = 0.0;  // fraction of shared top-k entries, in [0,1]
};

inline void check_equal_length(std::size_t a, std::size_t b) {
    if (a != b) {
        throw std::invalid_argument(
            "length mismatch: " + std::to_string(a) + " vs " + std::to_string(b));
    }
}

// Cosine similarity in [-1, 1]. Returns 1.0 for identical non-zero vectors,
// 0.0 for orthogonal vectors. Throws on length mismatch or all-zero inputs.
inline double cosine_similarity(const std::vector<float>& a,
                                const std::vector<float>& b) {
    check_equal_length(a.size(), b.size());
    double dot = 0.0, na2 = 0.0, nb2 = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na2  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb2  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na2 == 0.0 || nb2 == 0.0) {
        throw std::invalid_argument("cosine_similarity: zero vector");
    }
    return dot / (std::sqrt(na2) * std::sqrt(nb2));
}

inline double max_absolute_error(const std::vector<float>& a,
                                 const std::vector<float>& b) {
    check_equal_length(a.size(), b.size());
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(static_cast<double>(a[i]) -
                                  static_cast<double>(b[i])));
    }
    return m;
}

inline double rmse(const std::vector<float>& a, const std::vector<float>& b) {
    check_equal_length(a.size(), b.size());
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(a.size()));
}

// Top-k agreement: fraction of entries that appear in the top-k of both vectors.
// k == 0 means "all entries" (i.e. identity for equal-length, identical vectors).
inline double top_k_agreement(const std::vector<float>& a,
                              const std::vector<float>& b,
                              std::size_t k) {
    check_equal_length(a.size(), b.size());
    if (k == 0 || k > a.size()) k = a.size();
    auto top_indices = [](const std::vector<float>& v, std::size_t kk) {
        std::vector<std::size_t> idx(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](std::size_t x, std::size_t y) {
                              return v[x] > v[y];
                          });
        return std::vector<std::size_t>(idx.begin(), idx.begin() + kk);
    };
    const auto ta = top_indices(a, k);
    const auto tb = top_indices(b, k);
    // top_indices returns indices sorted by descending value, not by index;
    // std::set_intersection needs both inputs sorted ascending by element.
    auto sorted = [](std::vector<std::size_t> v) {
        std::sort(v.begin(), v.end());
        return v;
    };
    const auto sa = sorted(ta);
    const auto sb = sorted(tb);
    std::vector<std::size_t> intersect;
    intersect.reserve(std::min(sa.size(), sb.size()));
    std::set_intersection(sa.begin(), sa.end(), sb.begin(), sb.end(),
                          std::back_inserter(intersect));
    return static_cast<double>(intersect.size()) / static_cast<double>(k);
}

inline ComparisonResult compare(const std::vector<float>& a,
                                const std::vector<float>& b,
                                std::size_t top_k = 0) {
    ComparisonResult r;
    r.cosine_similarity = cosine_similarity(a, b);
    r.max_absolute_error = max_absolute_error(a, b);
    r.rmse = rmse(a, b);
    r.top_k_agreement = top_k_agreement(a, b, top_k);
    return r;
}

// Tolerance checks with the classes described in section 0.3 of
// docs/ARCHITECTURE_EVIDENCE.md.
inline void require_within(const ComparisonResult& r,
                           double min_cosine,
                           double max_rmse,
                           double max_abs,
                           double min_top_k_agreement = 1.0,
                           std::size_t top_k = 0) {
    if (r.cosine_similarity < min_cosine) {
        throw std::runtime_error(
            "cosine_similarity " + std::to_string(r.cosine_similarity) +
            " below threshold " + std::to_string(min_cosine));
    }
    if (r.rmse > max_rmse) {
        throw std::runtime_error(
            "rmse " + std::to_string(r.rmse) +
            " above threshold " + std::to_string(max_rmse));
    }
    if (r.max_absolute_error > max_abs) {
        throw std::runtime_error(
            "max_absolute_error " + std::to_string(r.max_absolute_error) +
            " above threshold " + std::to_string(max_abs));
    }
    if (r.top_k_agreement < min_top_k_agreement) {
        throw std::runtime_error(
            "top_k_agreement " + std::to_string(r.top_k_agreement) +
            " below threshold " + std::to_string(min_top_k_agreement) +
            " (top_k=" + std::to_string(top_k) + ")");
    }
}

} // namespace celeg::test::numerical
