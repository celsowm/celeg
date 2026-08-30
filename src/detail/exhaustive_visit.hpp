#pragma once


#include <type_traits>
#include <utility>
#include <variant>

namespace celeg {

template <typename... Handlers>
struct ExhaustiveHandlers : Handlers... {
    using Handlers::operator()...;
};

template <typename VariantRef, typename... Handlers>
decltype(auto) visit_exhaustive(VariantRef&& value, Handlers&&... handlers) {
    ExhaustiveHandlers<std::decay_t<Handlers>...> dispatch{
        std::forward<Handlers>(handlers)...};
    return std::visit(
        [&dispatch](auto&& alternative) -> decltype(auto) {
            return dispatch(&alternative);
        },
        std::forward<VariantRef>(value));
}

}
