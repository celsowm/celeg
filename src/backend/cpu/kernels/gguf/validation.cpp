#include "celeg/backend/cpu/gguf.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace celeg {

CpuLinearWeight CpuLinearWeight::from_q4(Q4GroupMatrix matrix) {
    matrix.validate();
    CpuLinearWeight result;
    result.rows = matrix.rows;
    result.cols = matrix.cols;
    result.segments.emplace_back(std::move(matrix));
    return result;
}

CpuLinearWeight CpuLinearWeight::from_ggml(GgmlMatrixView matrix) {
    matrix.validate();
    CpuLinearWeight result;
    result.rows = matrix.rows;
    result.cols = matrix.cols;
    result.segments.emplace_back(matrix);
    return result;
}

void CpuInt8Matrix::validate() const {
    if (rows == 0 || cols == 0 || !values || !scales ||
        values->size() != static_cast<size_t>(rows) * cols ||
        scales->size() != static_cast<size_t>(rows)) {
        throw std::runtime_error("invalid CPU INT8 matrix");
    }
}

CpuLinearWeight CpuLinearWeight::from_int8(CpuInt8Matrix matrix) {
    matrix.validate();
    CpuLinearWeight result;
    result.rows = matrix.rows;
    result.cols = matrix.cols;
    result.segments.emplace_back(std::move(matrix));
    return result;
}

size_t CpuLinearWeight::memory_bytes() const {
    size_t total = 0;
    for (const CpuLinearMatrix& segment : segments) {
        total += std::visit([](const auto& value) {
            return value.memory_bytes();
        }, segment);
    }
    return total;
}

bool CpuLinearWeight::gguf_native() const {
    return !segments.empty() &&
        std::all_of(segments.begin(), segments.end(), [](const CpuLinearMatrix& value) {
            return std::holds_alternative<GgmlMatrixView>(value);
        });
}

void CpuLinearWeight::validate() const {
    if (rows == 0 || cols == 0 || segments.empty()) {
        throw std::invalid_argument("invalid CPU linear weight rows=" +
            std::to_string(rows) + " cols=" + std::to_string(cols) +
            " segments=" + std::to_string(segments.size()));
    }
    size_t segment_rows = 0;
    for (const CpuLinearMatrix& segment : segments) {
        std::visit([&](const auto& value) {
            value.validate();
            if (value.cols != cols) {
                throw std::invalid_argument("CPU linear segment width mismatch");
            }
            segment_rows += value.rows;
        }, segment);
    }
    if (segment_rows != rows) {
        throw std::invalid_argument("CPU linear segment row count mismatch");
    }
}

}
