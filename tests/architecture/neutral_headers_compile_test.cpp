#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/definition.hpp"
#include "celeg/model/graph.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/model/runtime_types.hpp"
#include "celeg/model/session.hpp"
#include "celeg/runtime/cache/prefix_cache.hpp"
#include "celeg/serve/types.hpp"

int main() {
    celeg::GenerationConfig generation;
    celeg::SessionState session(generation);
    celeg::serve::GenerateRequest request;
    request.generation = generation;
    return session.phase_ == celeg::SessionPhase::Empty &&
                   request.generation.seed == generation.seed
               ? 0
               : 1;
}
