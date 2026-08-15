#pragma once

#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/weight_repository.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace celeg {

class GgufTensorResolver;

class GgufRepository final : public IWeightRepository,
                             public INativeBlockStorageRepository {
public:
    explicit GgufRepository(std::shared_ptr<GgufFile> gguf);
    ~GgufRepository();

    GgufRepository(const GgufRepository&) = delete;
    GgufRepository& operator=(const GgufRepository&) = delete;

    bool contains(std::string_view name) const override;
    HostTensorView tensor(std::string_view name) const override;
    std::vector<std::string> names() const override;
    bool has_native_block_storage() const override { return true; }

private:
    std::unique_ptr<GgufTensorResolver> tensors_;
};

}
