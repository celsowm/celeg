# AGENTS.md

## Hard rules

- **NO backward compatibility.** When refactoring, fully replace old access
  patterns with the new interface. Do not keep legacy shortcuts, `impl_->`
  member access, or shim helpers "for compatibility". Delete the old path.
- When a regex substitution touches multiple overlapping patterns, apply the
  most specific pattern first (e.g. `position_device_` before `position_`),
  otherwise a partial match corrupts the output (seen: `position_device_` ->
  `position()device_`).
- Prefer interface accessors over reaching into internals:
  - `IPackedSession` exposes `position()` / `set_position(int)`,
    `sampled_host()` / `set_sampled_host_value(int32_t)`, `rope_cos()` /
    `rope_sin()` (raw `const __nv_bfloat16*`, NOT a class — do not call
    `.data()` on them), `metrics()`, `phase()` / `set_phase(...)`,
    `active_segmented_attention()` / `set_active_segmented_attention(...)`,
    `use_segmented_attention(int)`, `max_context()`, `packed_session()`.
  - `++x.member` on a value-returning accessor is illegal; use
    `x.set_member(x.member() + 1)`.

## Build

```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cd /d D:\ia\lfm25-cuda-cpp && cmake --build build-cuda -j'
```

Clean build: append `--clean-first`. All 36 test executables live in
`build-cuda/` and must pass (`ALL PASS (36)`).

## Smoke checks

```powershell
./build-cuda/lfm25-run.exe --model ./model/LFM2.5-230M --prompt "Hello" --max-new-tokens 4
./build-cuda/lfm25-run.exe --model ./model/LFM2.5-1.2B-Instruct --prompt "Explain CUDA in one sentence." --max-new-tokens 16
./build-cuda/lfm25-concurrent-benchmark.exe ./model/LFM2.5-230M/model.safetensors ./model/LFM2.5-230M/tokenizer.json "Hello world" 4 8
```
