#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/resolved.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/// Shared fixtures for the automatic-inference test modules: an in-memory
/// weight repository plus one metadata/repository builder pair per
/// architecture family exercised by the suite.
namespace celeg::automatic_inference_test {

class MemoryRepository final : public celeg::IWeightRepository {
public:
    void add(std::string name, std::vector<int64_t> shape) {
        shapes_.emplace(std::move(name), std::move(shape));
    }

    bool contains(std::string_view name) const override {
        return shapes_.contains(std::string(name));
    }
    celeg::HostTensorView tensor(std::string_view name) const override {
        const auto it = shapes_.find(std::string(name));
        if (it == shapes_.end()) throw std::out_of_range("missing synthetic tensor");
        return {celeg::TensorDType::BF16, it->second, nullptr, 0};
    }
    std::vector<std::string> names() const override {
        std::vector<std::string> result;
        for (const auto& [name, shape] : shapes_) {
            (void)shape;
            result.push_back(name);
        }
        return result;
    }

private:
    std::unordered_map<std::string, std::vector<int64_t>> shapes_;
};

celeg::CheckpointMetadata metadata();
std::shared_ptr<MemoryRepository> repository();

celeg::CheckpointMetadata gguf_metadata();
std::shared_ptr<MemoryRepository> gguf_repository();

celeg::CheckpointMetadata no_rope_gguf_metadata();

celeg::CheckpointMetadata hybrid_gguf_metadata();
std::shared_ptr<MemoryRepository> hybrid_gguf_repository();

celeg::CheckpointMetadata ling_metadata();
std::shared_ptr<MemoryRepository> ling_repository();

celeg::CheckpointMetadata qwen35_metadata();
std::shared_ptr<MemoryRepository> qwen35_repository();

celeg::CheckpointMetadata agnes_metadata();
std::shared_ptr<MemoryRepository> agnes_repository(const std::string& layer_root,
                                                   const std::string& model_root);

void check_agnes_model(const celeg::ResolvedModel& model);

}
