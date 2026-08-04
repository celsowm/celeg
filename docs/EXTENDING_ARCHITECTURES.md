# Extending model architectures

An architecture extension is registered through an `IArchitecture` family
bundle. The resolution path is composed from the neutral stages declared in
[`architecture.hpp`](../include/celeg/model/architecture.hpp): probe/metadata
interpretation, topology, graph, weights, and capabilities/provenance.

Family-specific metadata decoders and naming policies belong under
`src/models/<family>/`. Generic model and backend headers must not include
family-specific types or checkpoint-format classes. Resolve tensor names and
expected shapes before invoking a CPU or CUDA compiler.

Add focused tests for each stage and a catalog-resolution test. If the family
has packed execution support, also provide per-layer dimension fixtures and a
backend-boundary test using a neutral/fake repository.
