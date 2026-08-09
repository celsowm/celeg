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

`ModelGraph` is the semantic owner of executable layer schedules: its
`LayerSpec` variants, normalization edges, primitive specs, and feed-forward
choices are what backend compilers consume. `RuntimeTopology` is a derived
runtime shape for allocation maxima, cache indexing, and compatibility checks;
new semantic decisions belong in the graph. The
`derive_runtime_topology_from_graph()` function is the synchronization boundary
for those derived tables.

When adding a primitive, keep the extension unit centered on the primitive:
add its graph spec and validation, primitive-owned weight requirements, then
the CPU/CUDA lowering and only the token, chunk, or packed executors it
supports. Numerical kernels may remain specialized, but all paths consume the
same graph-owned semantics and compiled policy. Packed CUDA operators join the
focused compatibility, metadata/workspace, layer-operator, and decode/prefill
components rather than adding another decision tree to the executor.
