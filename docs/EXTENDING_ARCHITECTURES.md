# Extending CELEG semantics

CELEG does not add support by naming a new checkpoint family. A checkpoint is
normalized into evidence, resolved into canonical semantic facts, and then
compiled by the backend-neutral graph pipeline:

```text
checkpoint -> evidence -> semantic facts -> model graph -> CPU/CUDA
```

If a new checkpoint uses semantics already represented by CELEG, no source
change should be necessary. Identity strings are useful for provenance and
diagnostics, but never select runtime behavior.

## Adding a reusable capability

Choose the narrowest semantic extension that is actually missing:

- add a metadata inference rule when a source convention exposes an existing
  semantic fact;
- add a tensor naming grammar when a new spelling maps to an existing tensor
  role;
- add a tokenizer behavior when tokenizer graph evidence requires a new
  pre-tokenization or normalization operation;
- add a `ChatTemplateProgram` instruction when a supported interaction
  construct is missing;
- add a `ToolCallGrammar` primitive for a new wire protocol;
- add a vision operation or `VisionPipelineSpec` field for a genuinely new
  image transformation;
- add a checkpoint/config importer when format-specific normalization is
  required.

Keep import, solving, execution, and composition separate. Backends consume
semantic requirements and capabilities; they must not inspect repository
names, `model_type`, architecture IDs, chat profiles, or vision labels.

## Validation checklist

Every semantic extension should include focused tests for evidence, canonical
resolution, and the relevant protocol or numerical behavior. Also add:

1. an unknown-identity clone test that changes only provenance strings;
2. a poisoning test that supplies a misleading identity while preserving
   structural evidence;
3. an explainability assertion showing the accepted evidence and fingerprint;
4. CPU and CUDA parity where the capability is executable by both backends.

Unsupported or ambiguous semantics must fail explicitly at load time. Do not
add a family registry, a compatibility shim, or a fallback selected by model
name.
