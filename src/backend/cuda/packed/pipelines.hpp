#pragma once

#include "celeg/backend/cuda/packed/executor.hpp"

namespace celeg {

struct PackedDecodeExecutorImpl;

class PackedDecodePipeline {
public:
    explicit PackedDecodePipeline(PackedDecodeExecutorImpl& state);
    void run(const std::vector<PackedSessionContext>& sessions,
             const std::vector<std::vector<uint32_t>>* page_tables,
             std::span<int32_t> output);

private:
    PackedDecodeExecutorImpl* state_;
};

class PackedPrefillPipeline {
public:
    explicit PackedPrefillPipeline(PackedDecodeExecutorImpl& state);
    void run(const std::vector<PackedSessionContext>& sessions,
             const std::vector<std::vector<uint32_t>>* page_tables,
             const std::vector<int32_t>& tokens,
             const std::vector<PackedPrefillRow>& rows);

private:
    PackedDecodeExecutorImpl* state_;
};

}
