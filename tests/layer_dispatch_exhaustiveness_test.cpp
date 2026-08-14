// Mixer dispatch must be exhaustive at compile time (SOLID review §2.1).
//
// The real `celeg::Layer` variant lives in a CUDA implementation header and
// cannot be extended from a test, so this test exercises the shared dispatch
// mechanism (`celeg::visit_exhaustive`, used by `visit_layer` on the CUDA side
// and `visit_operator_weights` on the CPU side) against a local variant that
// mirrors the five current mixer alternatives, plus a variant that adds a sixth
// one. The point of the test is the static assertions: a handler set that does
// not name every alternative is not invocable for the missing alternative, so
// `visit_exhaustive` fails to compile at that dispatch site.

#include "celeg/detail/exhaustive_visit.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

namespace {

struct Attention {};
struct Convolution {};
struct GatedDeltaNet {};
struct Mamba2 {};
struct MlpOnly {};
// Stands in for a sixth mixer added to the dispatch domain later.
struct FutureMixer {};

using MirrorLayer =
    std::variant<Attention, Convolution, GatedDeltaNet, Mamba2, MlpOnly>;
using ExtendedLayer = std::variant<Attention, Convolution, GatedDeltaNet,
                                   Mamba2, MlpOnly, FutureMixer>;

struct RecordAttention {
    std::string* seen;
    void operator()(const Attention*) const { *seen = "attention"; }
};
struct RecordConvolution {
    std::string* seen;
    void operator()(const Convolution*) const { *seen = "convolution"; }
};
struct RecordGatedDeltaNet {
    std::string* seen;
    void operator()(const GatedDeltaNet*) const { *seen = "gated_delta_net"; }
};
struct RecordMamba2 {
    std::string* seen;
    void operator()(const Mamba2*) const { *seen = "mamba2"; }
};
struct RecordMlpOnly {
    std::string* seen;
    void operator()(const MlpOnly*) const { *seen = "mlp_only"; }
};

using CurrentHandlers =
    celeg::ExhaustiveHandlers<RecordAttention, RecordConvolution,
                              RecordGatedDeltaNet, RecordMamba2, RecordMlpOnly>;
using IncompleteHandlers =
    celeg::ExhaustiveHandlers<RecordAttention, RecordConvolution,
                              RecordGatedDeltaNet, RecordMamba2>;

// Every alternative of the current dispatch domain is named by a handler.
static_assert(std::is_invocable_v<CurrentHandlers, const Attention*>);
static_assert(std::is_invocable_v<CurrentHandlers, const Convolution*>);
static_assert(std::is_invocable_v<CurrentHandlers, const GatedDeltaNet*>);
static_assert(std::is_invocable_v<CurrentHandlers, const Mamba2*>);
static_assert(std::is_invocable_v<CurrentHandlers, const MlpOnly*>);

// A handler set that forgets an alternative cannot dispatch it: the
// corresponding `visit_exhaustive` call would not compile.
static_assert(!std::is_invocable_v<IncompleteHandlers, const MlpOnly*>);

// Adding a sixth alternative breaks the existing handler set the same way,
// which is what forces every dispatch site to be revisited when a mixer is
// added. Writing
//     celeg::visit_exhaustive(extended, RecordAttention{&seen}, ...);
// with only the five handlers below would not compile.
static_assert(std::variant_size_v<ExtendedLayer> ==
              std::variant_size_v<MirrorLayer> + 1);
static_assert(!std::is_invocable_v<CurrentHandlers, const FutureMixer*>);

// There is deliberately no generic fallback in the mechanism: a handler set is
// invocable only for the alternatives it names.
struct Unrelated {};
static_assert(!std::is_invocable_v<CurrentHandlers, const Unrelated*>);

std::string dispatch(const MirrorLayer& layer) {
    std::string seen;
    celeg::visit_exhaustive(layer, RecordAttention{&seen}, RecordConvolution{&seen},
                            RecordGatedDeltaNet{&seen}, RecordMamba2{&seen},
                            RecordMlpOnly{&seen});
    return seen;
}

} // namespace

int main() try {
    CELEG_TEST_CHECK(dispatch(MirrorLayer{Attention{}}) == "attention");
    CELEG_TEST_CHECK(dispatch(MirrorLayer{Convolution{}}) == "convolution");
    CELEG_TEST_CHECK(dispatch(MirrorLayer{GatedDeltaNet{}}) == "gated_delta_net");
    CELEG_TEST_CHECK(dispatch(MirrorLayer{Mamba2{}}) == "mamba2");
    CELEG_TEST_CHECK(dispatch(MirrorLayer{MlpOnly{}}) == "mlp_only");

    // Lambda handlers, the form used by the backend dispatch sites.
    int convolutions = 0;
    for (const MirrorLayer& layer :
         {MirrorLayer{Convolution{}}, MirrorLayer{Attention{}}}) {
        celeg::visit_exhaustive(layer,
          [&](const Attention*) {},
          [&](const Convolution*) { ++convolutions; },
          [&](const GatedDeltaNet*) {},
          [&](const Mamba2*) {},
          [&](const MlpOnly*) {});
    }
    CELEG_TEST_CHECK(convolutions == 1);

    std::cout << "layer_dispatch_exhaustiveness_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "layer_dispatch_exhaustiveness_test failed: " << error.what() << '\n';
    return 1;
}
