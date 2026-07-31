# Runtime Architecture

The runtime is divided into four dependency directions:

```text
checkpoint formats -> model definitions -> runtime scheduling -> backend operators
                                      \-> architecture programs
```

Public model definitions and checkpoint contracts are backend-neutral. CUDA
headers live under `lfm/backend/cuda` and are intentionally absent from the
installed public header manifest. The CUDA model implementation owns device
allocation and forwards compact operation contexts to the packed executor;
public `Model` does not expose that executor or its context type.

## Architecture providers

`IArchitectureProvider` converts validated checkpoint metadata into a common
`ModelDefinition`. Providers own architecture detection and tensor naming;
operators and schedulers consume common dimensions and operation arguments.
`ArchitectureRegistry` is the single selection point. LFM2 and Granite are
registered providers. Granite supplies validated configuration, standard
Safetensors tensor naming, explicit numerical modifiers, and dense CPU/CUDA
execution through the same backend-neutral operator contracts; it does not add
an architecture switch to backend kernels.

## Weight pipeline

`IWeightRepository` exposes source tensors. Optional location and random-access
operations are separate capabilities. `TensorResolver` maps semantic
`TensorRole` values to source names and validates shapes before the CUDA
`WeightLoader` materializes storage. Source encoding and runtime encoding are
therefore decisions at different boundaries.

## Session lifecycle

The public facade exposes focused session, diagnostics, and persistence views.
Packed decode receives a `PackedSessionContext` snapshot assembled by the CUDA
backend-internal factory. It contains operation-specific bindings and callbacks,
not a general model interface or arbitrary implementation access.

## Adding a backend

Add backend operators under `src/backend/<backend>`, keep backend types out of
`lfm/model` and `lfm/checkpoint`, and register only the backend implementation
at model construction. The generic scheduler, checkpoint capabilities, and
sampling contracts should remain unchanged.
