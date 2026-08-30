#pragma once

#include "celeg/checkpoint/tensor.hpp"
#include "celeg/checkpoint/tokenizer.hpp"

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

class IWeightRepository {
public:
    virtual ~IWeightRepository() = default;
    virtual bool contains(std::string_view name) const = 0;
    virtual HostTensorView tensor(std::string_view name) const = 0;
    virtual std::vector<std::string> names() const = 0;
};

class ILocatableTensorRepository {
public:
    virtual ~ILocatableTensorRepository() = default;
    virtual TensorLocator locate(std::string_view name) const = 0;
};

class IRandomAccessTensorReader {
public:
    virtual ~IRandomAccessTensorReader() = default;
    virtual void read(const TensorLocator& locator,
                      std::span<std::byte> destination) const = 0;
};

class INativeBlockStorageRepository {
public:
    virtual ~INativeBlockStorageRepository() = default;
    virtual bool has_native_block_storage() const = 0;
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

inline const IRandomAccessTensorReader&
require_random_access_tensor_reader(const IWeightRepository& repository) {
    const auto* reader = dynamic_cast<const IRandomAccessTensorReader*>(&repository);
    if (reader == nullptr) {
        throw std::runtime_error(
            "checkpoint repository does not provide random-access tensor reads");
    }
    return *reader;
}

class ITokenizerDataRepository {
public:
    virtual ~ITokenizerDataRepository() = default;
    virtual TokenizerData tokenizer_data() const = 0;
};

}
