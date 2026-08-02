#include "celeg/model/weights/roles.hpp"
#include "support/assertions.hpp"

#include <cstddef>
#include <unordered_map>

namespace {
class MemoryRepository final : public celeg::IWeightRepository {
public:
    bool contains(std::string_view name) const override {
        return tensors.contains(std::string(name));
    }
    celeg::HostTensorView tensor(std::string_view name) const override {
        return tensors.at(std::string(name));
    }
    std::vector<std::string> names() const override {
        std::vector<std::string> result;
        for (const auto& [name, unused] : tensors) { (void)unused; result.push_back(name); }
        return result;
    }
    std::unordered_map<std::string, celeg::HostTensorView> tensors;
};
}

int main() {
    MemoryRepository repository;
    repository.tensors.emplace("model.layers.0.self_attn.q_proj.weight",
        celeg::HostTensorView{celeg::TensorDType::BF16, {2, 4}, nullptr, 16});
    celeg::CelegTensorNamingPolicy policy;
    celeg::TensorResolver resolver(repository, policy);
    const celeg::ResolvedTensor tensor = resolver.resolve({
        celeg::TensorRole::AttentionQuery, 0, -1, {2, 4}});
    CELEG_TEST_CHECK(tensor.source_name == "model.layers.0.self_attn.q_proj.weight");
    CELEG_TEST_CHECK(tensor.view.shape == std::vector<int64_t>({2, 4}));
    return 0;
}
