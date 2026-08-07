# Qwen3.5 support plan

This plan is anchored on the current official `Qwen/Qwen3.5-35B-A3B` checkpoint,
not on an older architecture selected merely because it is easier to support.
The target runtime is CUDA on a 12 GB RTX 3060 with expert offload to host or
disk-backed storage.

## What the checkpoint actually requires

- 40 language layers: 30 GatedDeltaNet linear-attention layers and 10 full
  attention layers.
- 256 routed experts, top-8 routing, plus a shared expert.
- BF16 base weights, with approximately 72 GB across the official shards.
- M-RoPE and a separate vision tower for the multimodal checkpoint family.
- One MTP layer (`mtp_num_hidden_layers=1`) with its own embedding, fusion
  projection, full-attention decoder layer, norm and LM head.

## Implementation order

1. **GatedDeltaNet first.** Establish numerical parity for convolution state,
   recurrent delta state, alpha/beta gates, RMSNorm, decode, ragged prefill and
   packed execution. The recurrent state must remain session-local.
2. **General MoE.** Separate router semantics, shared-expert semantics and
   expert storage from the Qwen naming policy. Then cover CPU/CUDA, resident
   experts, host cache, disk source, LRU/LFU policy and packed routing.
3. **3060 offload validation.** Keep the GPU cache bounded by the routed top-K,
   promote experts asynchronously where safe, and use tokenwise disk-backed
   prefill when a prompt would otherwise require too many distinct experts at
   once. Measure correctness first; the current disk-backed smoke is a stress
   test, not a performance claim.
4. **Vision and M-RoPE.** Load the official vision tower, interpolate learned
   positional embeddings, expand image markers into visual tokens and carry
   all three rotary coordinates through CUDA execution.
5. **MTP/speculation (experimental opt-in).** The official `mtp.*` weights,
   one full-attention auxiliary layer, its own KV cache and the synthetic MoE
   residency entry now load and execute with `--mtp`, including disk-backed
   expert storage. The path is deliberately restricted to one MTP layer,
   local KV, non-paged execution, `--no-cuda-graph` and exactly one speculative
   token. In greedy decoding,
   an accepted one-token proposal is consumed on the next step and the target
   continues verifying the stream; rejected proposals fall back to the normal
   sampler. The transaction boundary also snapshots candidate state and MTP
   logits. Multi-token acceptance, stochastic sampling and a speed claim are
   still not advertised.
6. **Only then add more activations or norms.** Current concrete Qwen3.5 needs
   SwiGLU, GELU-tanh in vision, and RMSNorm with the model's `+1` convention.
   No speculative activation/norm abstraction is justified until another
   current target model demonstrates a real requirement.

## Acceptance gates

- Architecture resolution tests cover the official metadata, including MTP.
- Full CUDA CTest remains green.
- A real Qwen3.5 BF16 run succeeds through disk-backed expert offload on the
  reference GPU.
- Numerical comparison covers GDN stateful decode versus a trusted reference.
- `--mtp` loads and executes the official predictor, but it is not advertised
  as an acceleration until candidate acceptance, rejection and repeated decode
  produce the same committed state as ordinary decoding.

The implementation deliberately does not claim MTP acceleration merely because
the checkpoint contains `mtp.*` tensors. The official vLLM predictor uses a
separate fusion projection and full-attention decoder layer, and its own
documentation calls MTP a speculative-decoding method. Qwen3.5's recurrent
DeltaNet layers make rollback part of the semantic contract, so the current
opt-in path is a loading/execution milestone, not a performance claim.

Official references:

- https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- https://huggingface.co/docs/transformers/model_doc/qwen3_5
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py
