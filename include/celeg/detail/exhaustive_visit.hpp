#pragma once

// Exhaustive variant dispatch.
//
// `visit_exhaustive` builds an overload set from the supplied handlers and
// visits the variant with it. There is deliberately no generic (`auto`)
// fallback: every alternative must be matched by a handler that names its
// concrete type, so adding an alternative to a dispatch domain turns every
// call site that has not been updated into a compile error instead of a
// silently unreachable branch.
//
// Handlers receive a pointer to the active alternative. The pointer is never
// null; the pointer form exists so that call sites keep the same variable
// shape as the `std::get_if` ladders this mechanism replaces.

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

} // namespace celeg
