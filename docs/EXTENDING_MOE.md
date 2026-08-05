# Extending MoE families

MoE family code resolves checkpoint metadata and tensor names into the
model-neutral `MoeLayerProgram` carried by `CompiledModelProgram`. The program
contains routing semantics, expert MLP dimensions, payload regions, shared
expert combination order, residency requirements, and a compatibility
fingerprint. It contains no checkpoint tensor names or architecture identity.

The extension path is:

1. Add family-owned metadata decoding and a naming policy.
2. Populate `MixtureOfExpertsSpec` for each MoE layer, including layer-local
   expert count, top-K, intermediate width, bias/normalization/scaling, and
   grouped/shared semantics when applicable.
3. Resolve family tensor requests and build the normal `ResolvedModel`.
4. Let `build_model_program()` validate and fingerprint the semantic program.
5. Add backend lowering only when the new semantic operation is supported;
   unsupported grouped, shared, or payload-layout operations must be rejected
   during compilation.

CPU and CUDA execution consume the compiled program. Expert caches and
residency use `ExpertKey` plus payload manifests; they do not inspect family
names or checkpoint suffixes. A new compatible family therefore requires
family-owned resolution and registration, not changes to generic routing,
cache, residency, packed execution, or the C API.

The current built-in LFM2 family is the real routed top-K example. The
compiler tests also construct a synthetic grouped program with independently
located payload regions and verify that semantic fields change its
fingerprint and overlapping regions are rejected.
