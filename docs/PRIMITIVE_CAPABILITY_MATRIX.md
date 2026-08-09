# Primitive capability matrix

This matrix records implementation evidence for the model-agnostic primitive
boundary. A cell is marked complete only when the corresponding lowering and
test exist; a semantic type alone is not evidence of backend support.

| Primitive or policy | IR | CPU token/chunk | CUDA token/chunk | Paged state | Evidence |
| --- | --- | --- | --- | --- | --- |
| Full causal MHA/GQA/MQA | complete | complete | complete | complete | CPU/CUDA attention tests |
| Sliding-window attention | complete | complete | complete | complete | CPU/CUDA paged paths |
| Bidirectional/prefix/sparse patterns | complete | reference | pending | CPU reference | `cpu_paged_kv_test` |
| Partial rotary | complete | complete | complete | n/a | RoPE kernel tests |
| Split-half / adjacent-pair RoPE | complete | complete | complete | n/a | `attention_semantics_test`, packed per-row RoPE lowering |
| M-RoPE | complete | complete | complete | n/a | `cpu_mrope_test`, CUDA M-RoPE path |
| Linear, Dynamic NTK, YaRN RoPE | complete | complete | complete | n/a | position/compiler tests |
| LongRoPE | complete | complete | complete | n/a | CPU/CUDA lowering and compiler tests |
| Llama-3 frequency scaling | complete | complete | complete | n/a | position/compiler tests |
| Attention output transform: orthogonalize against current V | complete | token native; chunk/packed exact fallback | token + packed native | packed paged native; direct legacy path guarded | `attention_semantics_test`, GPT-X2.5 architecture-resolution test |
| ALiBi | complete | complete | complete | n/a | CPU paged attention and CUDA kernel lowering/test |
| Relative position bias | complete | complete | pending | n/a | CPU paged attention and weight-plan tests |
| Latent attention state | complete | complete | pending | CPU latent pages | CPU latent reference path |
| External attention memory | complete | complete | pending | CPU external pages | CPU external-memory path |
| Recurrent/linear attention | complete | token | sequential adapter | recurrent | recurrent operator tests |
| State storage policies | complete | layout metadata | layout metadata | partial | compiled state-layout validation |
| Streaming sink/recent cache | partial | pending | pending | pending | no end-to-end lowering yet |
| Context-parallel lowering | boundary only | n/a | pending | pending | backend extension boundary |

The remaining pending CUDA cells are intentional tracking entries: CUDA
compilation must not accept a semantic primitive until its cache layout, kernel
lowering, and parity tests are implemented.
