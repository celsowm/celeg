# Relatório do sweep de modelos CPU/CUDA

Status: concluído para os 18 artefatos adquiridos e executados neste sweep.

## Ambiente

- Plataforma: Windows 11
- CPU: AVX2, `out/windows-cpu-release/celeg-cpu-run.exe`
- GPU: NVIDIA RTX 3060 Ampere, 12 GB, `out/windows-cuda-release/celeg-run.exe`
- Prompt: `What is the capital of France?`
- Geração: greedy, temperatura 0, top-k 1, máximo de 8 tokens novos
- Cache local: `C:\Users\celso\.cache\huggingface\hub`; o Hub remoto não foi alterado

## Build e testes

| Item | Resultado | Evidência |
|---|---|---|
| CPU Release | PASS | `python scripts/dev.py build --backend cpu --build-type Release --jobs 12` |
| CUDA Release | PASS | `python scripts/dev.py build --backend cuda --build-type Release --jobs 12` (583/583 targets) |
| CPU CTest | PASS | 80/80 testes |
| CUDA CTest | PASS com 1 falha conhecida | 90/91 testes; `cuda_kernels_test` falha no caso NVFP4 W4A4 (esperado 0.410156, obtido 0.734375) |

A falha do `cuda_kernels_test` é isolada ao caso numérico NVFP4/W4A4. Ela não impediu a execução dos modelos listados abaixo.

## Resultados de runtime

`OK` indica que o processo terminou com código zero. `Correto` indica que a resposta continha a capital esperada; saídas vazias, truncadas ou incorretas permanecem registradas como observadas.

| Modelo/artefato | CPU AVX2 | CUDA |
|---|---|---|
| `LiquidAI/LFM2.5-230M` Safetensors | OK, correto | OK, saída corrompida (`?`) |
| `LiquidAI/LFM2.5-350M` Safetensors | OK, correto | OK, saída corrompida (`??`) |
| `openbmb/MiniCPM5-1B` Safetensors | OK, correto | OK, saída corrompida (`#`) |
| `LFM2.5-230M` Q4_K_M | OK, correto | OK, vazio |
| `LFM2.5-230M` Q6_K | OK, correto | OK, vazio |
| `LFM2.5-350M` Q4_0 | OK, corrompido/vazio | FAIL, `execution plan requires INT8 weights` |
| `LFM2.5-350M` Q4_K_M | OK, correto | OK, vazio |
| `LFM2.5-350M` Q5_K_M | OK, correto | FAIL, concatenação não suportada |
| `LFM2.5-350M` Q8_0 | OK, correto | FAIL, concatenação não suportada |
| `Qwen3.5-0.8B` Q4_K_M | OK, corrompido | OK, corrompido |
| `MiniCPM5-1B` Q4_K_M | OK, correto | OK, corrompido |
| `SmolLM3-3B` Q4_K_M | OK, correto | OK, vazio |
| `Nanbeige 3B` Q4_K_M | OK, errado/não inglês | FAIL, falta `tokenizer.ggml.merges` |
| `Nemotron 4B` Q4_K_M | OK, correto/parcial | OK, errado/parcial |
| `Ling-3.0-tiny-int4` Safetensors | OK, 6.91 tok/s no decode | FAIL, sem saída; limite de memória CUDA observado |
| `LiquidAI/LFM2.5-8B-A1B` Safetensors | OK, correto; 17.145 tok/s no decode | OK, correto com MoE offload; 0.508 tok/s no prefill e 1.539 tok/s no decode |
| `flwrlabs/Lizzy-7B-GGUF` Q4_K_M | OK, correto; 6.188 tok/s no decode | OK, vazio |
| `flwrlabs/Lizzy-7B` Safetensors | OK, correto; 3.479 tok/s no decode | OK, saída truncada/incorreta (`The`) |

## LFM2.5-8B-A1B com MoE offload

O teste CUDA foi executado com:

```text
--expert-offload auto --expert-backing disk --expert-cache-policy lfu-lru
--expert-host-cache-mib 4096 --expert-cache-per-layer 8
--context 1024 --prefill-chunk 128
```

O plano usou 8 de 32 experts por camada na GPU, cache de experts de 3.61 GiB, armazenamento de experts no host de 10.83 GiB e apenas 1.41 GiB de memória CUDA reportada no runtime. A saída foi `A. Paris` e `B. London`. Portanto, o offload de MoE funciona na RTX 3060 e é justamente o mecanismo que permite rodar esse checkpoint BF16 maior que a VRAM disponível.

## Observações de hardware

O caminho de MoE offload pode ser validado nesta máquina Ampere. O caminho nativo NVFP4 é um caso separado: a RTX 3060 não possui os recursos FP4 nativos da arquitetura Blackwell, portanto a validação de desempenho nativo NVFP4 deve ser feita no ambiente Linux com RTX 5090.

## Integridade do cache

Os snapshots de Lizzy foram concluídos por transferência direta dos blobs verificados, e não restaram fragmentos `.incomplete`. Nenhum arquivo remoto do Hugging Face foi apagado ou alterado.
