# Runtime Architecture

The runtime is divided into four dependency directions:

```text
checkpoint formats -> model definitions -> runtime scheduling -> backend operators
                                      \-> architecture programs
```

The complete composition chain is:

```text
checkpoint format plugin -> architecture resolver -> validated model graph
-> backend compiler -> chat profile -> chat template -> tool-call codec
-> protocol adapter
```

Public model definitions and checkpoint contracts are backend-neutral. CUDA
headers live under `celeg/backend/cuda` and are intentionally absent from the
installed public header manifest. The CUDA model implementation owns device
allocation and forwards compact operation contexts to the packed executor;
public `Model` does not expose that executor or its context type.

## Architecture catalog

The composition root reads a `CheckpointMetadata`/`CheckpointView`, selects one
entry from the frozen `ArchitectureCatalog`, and asks its `IArchitecture` to
produce a complete `ResolvedModel`. Resolution owns detection, profile patches,
topology validation, `ModelGraph` construction, semantic `WeightPlan` creation,
tensor bindings, chat profile, and the model fingerprint. CPU and CUDA receive
only that resolved data and compile the graph into executable layer programs.

## Weight pipeline

`IWeightRepository` exposes source tensors. Optional location and random-access
operations are separate capabilities. Architecture modules map semantic
`TensorRole` values to source names and produce a shape-checked `WeightPlan`.
CPU/CUDA then split planning, materialization, expert residency, caching, and
offload without rediscovering the checkpoint architecture.

## Session lifecycle

The public facade exposes focused session, diagnostics, and persistence views.
Packed decode receives a `PackedSessionContext` snapshot assembled by the CUDA
backend-internal factory. It contains operation-specific bindings and callbacks,
not a general model interface or arbitrary implementation access.

## Adding a backend

Add backend operators under `src/backend/<backend>`, keep backend types out of
`celeg/model` and `celeg/checkpoint`, and register only the backend implementation
at model construction. The generic scheduler, checkpoint capabilities, and
sampling contracts should remain unchanged.

## Native tool-call evidence

The LFM2 codec follows `.externals/llama.cpp/models/templates/LFM2.5-8B-A1B.jinja`:
tool declarations use the tool-list markers, assistant calls use
`<|tool_call_start|>...[name(args)]<|tool_call_end|>`, and tool results use the
corresponding response markers. The Gemma 4 codec follows
`.externals/llama.cpp/models/templates/google-gemma-4-31B-it.jinja`: assistant
calls use `<|tool_call>call:name{json}<tool_call|>` and results use
`<|tool_response>response:name{...}<tool_response|>`. Both codecs preserve
arguments as JSON strings at the protocol boundary and assign stable call IDs
when the model format does not emit one. Granite is not advertised until its
marker vocabulary and parser are verified together.
