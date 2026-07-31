#include "lfm/model/weights/roles.hpp"

#include <cassert>
#include <cstddef>
#include <unordered_map>

using namespace lfm;

namespace {
class MemoryRepository final : public IWeightRepository {
public:
    bool contains(std::string_view name) const override {
        return tensors.contains(std::string(name));
    }
    HostTensorView tensor(std::string_view name) const override {
        return tensors.at(std::string(name));
    }
    std::vector<std::string> names() const override {
        std::vector<std::string> result;
        for (const auto& [name, unused] : tensors) { (void)unused; result.push_back(name); }
        return result;
    }
    std::unordered_map<std::string, HostTensorView> tensors;
};
}

int main() {
    MemoryRepository repository;
    repository.tensors.emplace("model.layers.0.self_attn.q_proj.weight",
        HostTensorView{TensorDType::BF16, {2, 4}, nullptr, 16});
    LfmTensorNamingPolicy policy;
    TensorResolver resolver(repository, policy);
    const ResolvedTensor tensor = resolver.resolve({
        TensorRole::AttentionQuery, 0, -1, {2, 4}});
    assert(tensor.source_name == "model.layers.0.self_attn.q_proj.weight");
    assert(tensor.view.shape == std::vector<int64_t>({2, 4}));
    return 0;
}
