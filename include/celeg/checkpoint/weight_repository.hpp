#pragma once

#include "celeg/checkpoint/tensor.hpp"

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

// Abstract weight source. Both the safetensors repository and the GGUF
// repository implement this so the weight loader and model builder stay
// format-agnostic. Implementations translate their native tensor naming to the
// canonical HuggingFace `model.*` scheme used by the model builder.
class IWeightRepository {
public:
    virtual ~IWeightRepository() = default;
    virtual bool contains(std::string_view name) const = 0;
    virtual HostTensorView tensor(std::string_view name) const = 0;
    virtual std::vector<std::string> names() const = 0;
};

// Optional checkpoint capability for repositories that can resolve a tensor
// to its source location. Repositories that only expose mapped tensor views
// implement IWeightRepository without pretending to support this operation.
class ILocatableTensorRepository {
public:
    virtual ~ILocatableTensorRepository() = default;
    virtual TensorLocator locate(std::string_view name) const = 0;
};

// Optional checkpoint capability for repositories that can copy source bytes
// by location. This is intentionally separate from IWeightRepository: GGUF
// and in-memory repositories do not need to inherit an operation that is
// invalid for their storage model.
class IRandomAccessTensorReader {
public:
    virtual ~IRandomAccessTensorReader() = default;
    virtual void read(const TensorLocator& locator,
                      std::span<std::byte> destination) const = 0;
};

inline const ILocatableTensorRepository&
require_locatable_tensor_repository(const IWeightRepository& repository) {
    const auto* locator = dynamic_cast<const ILocatableTensorRepository*>(&repository);
    if (locator == nullptr) {
        throw std::runtime_error(
            "checkpoint repository does not provide tensor locations");
    }
    return *locator;
}

// Capability discovery helper for cold/offloaded reads. The exception is
// raised at the boundary where the capability is required, rather than by a
// misleading default implementation on every repository.
inline const IRandomAccessTensorReader&
require_random_access_tensor_reader(const IWeightRepository& repository) {
    const auto* reader = dynamic_cast<const IRandomAccessTensorReader*>(&repository);
    if (reader == nullptr) {
        throw std::runtime_error(
            "checkpoint repository does not provide random-access tensor reads");
    }
    return *reader;
}

} // namespace celeg
