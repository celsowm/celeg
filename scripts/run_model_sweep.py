#!/usr/bin/env python3
"""
Run all resolvable HF-cache models on CPU and CUDA backends.
Classifies each: OK / FAIL / TIMEOUT / OOM.
Writes results to /home/IN.PGE.RJ.GOV.BR/fontesc/celeg/model_sweep_results.json
"""
import subprocess, json, time, os, sys
from datetime import datetime

REPO_LIST = [
    # (repo_id, weight_type)
    ("LiquidAI/LFM2.5-230M", "safetensors"),
    ("LiquidAI/LFM2.5-350M", "safetensors"),
    ("LiquidAI/LFM2.5-VL-450M", "safetensors"),
    ("google/gemma-4-E4B", "safetensors"),
    ("ibm-granite/granite-4.1-3b", "safetensors"),
    ("inclusionAI/Ling-3.0-tiny", "safetensors"),
    ("openbmb/MiniCPM5-1B", "safetensors"),
    ("Nanbeige/Nanbeige4.2-3B", "safetensors"),
    ("flwrlabs/Lizzy-7B", "safetensors"),
    ("LiquidAI/LFM2.5-230M-GGUF", "gguf"),
    ("LiquidAI/LFM2.5-350M-GGUF", "gguf"),
    ("flwrlabs/Lizzy-7B-GGUF", "gguf"),
    ("bartowski/Nanbeige_Nanbeige4.2-3B-GGUF", "gguf"),
    ("openbmb/MiniCPM5-1B-GGUF", "gguf"),
]

PROMPT = "What is the capital of France?"
MAX_TOKENS = 20
TEMP = 0.0
TOP_K = 1

CPU_RUN = "./out/linux-cpu-relwithdebinfo/celeg-cpu-run"
CUDA_RUN = "./out/linux-cuda-relwithdebinfo/celeg-run"

def run_model(run_cmd, repo, backend, extra_args=None):
    """Run a model and return (success, output, elapsed_s, error_type)."""
    cmd = [run_cmd, "--repo", repo, "--prompt", PROMPT,
           "--max-new-tokens", str(MAX_TOKENS),
           "--temperature", str(TEMP), "--top-k", str(TOP_K)]
    if extra_args:
        cmd.extend(extra_args)
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd="/home/IN.PGE.RJ.GOV.BR/fontesc/celeg")
        elapsed = time.time() - start
        stdout = proc.stdout
        stderr = proc.stderr
        combined = stdout + stderr
        if proc.returncode == 0 and "chat.template=" in combined:
            # Extract the generated text (last line usually)
            lines = combined.strip().split("\n")
            generated = lines[-1] if lines else ""
            return (True, generated.strip(), elapsed, None)
        else:
            # Classify error
            err = combined.strip().split("\n")[0] if combined.strip() else "unknown"
            if "cublasCreate" in combined:
                error_type = "cuda_init_failed"
            elif "RoPE" in combined and "rope_theta" in combined:
                error_type = "missing_rope_theta"
            elif "feed-forward normalization" in combined:
                error_type = "ff_norm_mismatch"
            elif "BPE produced token absent" in combined:
                error_type = "tokenizer_vocab_mismatch"
            elif "no chat template" in combined:
                error_type = "missing_chat_template"
            elif "out of memory" in combined.lower() or "OOM" in combined:
                error_type = "oom"
            elif "Killed" in combined:
                error_type = "oom_killed"
            else:
                error_type = "other"
            return (False, err, elapsed, error_type)
    except subprocess.TimeoutExpired:
        return (False, "timeout", 300, "timeout")
    except Exception as e:
        return (False, str(e), time.time() - start, "exception")

def main():
    results = {"timestamp": datetime.now().isoformat(),
               "prompt": PROMPT,
               "models": []}

    for repo, wtype in REPO_LIST:
        print(f"\n{'='*60}")
        print(f"MODEL: {repo} ({wtype})")
        print(f"{'='*60}")

        # CUDA run
        print(f"  CUDA:  ", end="", flush=True)
        cuda_ok, cuda_out, cuda_time, cuda_err = run_model(CUDA_RUN, repo, "cuda")
        cuda_status = "OK" if cuda_ok else f"FAIL({cuda_err})"
        print(cuda_status)
        if cuda_ok:
            print(f"    output: {cuda_out[:80]}")
        else:
            print(f"    error:  {cuda_out[:120]}")
        print(f"    time:   {cuda_time:.1f}s")

        # CPU run (only if CUDA worked, or always for comparison)
        print(f"  CPU:   ", end="", flush=True)
        cpu_ok, cpu_out, cpu_time, cpu_err = run_model(CPU_RUN, repo, "cpu")
        cpu_status = "OK" if cpu_ok else f"FAIL({cpu_err})"
        print(cpu_status)
        if cpu_ok:
            print(f"    output: {cpu_out[:80]}")
        else:
            print(f"    error:  {cpu_out[:120]}")
        print(f"    time:   {cpu_time:.1f}s")

        results["models"].append({
            "repo": repo,
            "weight_type": wtype,
            "cuda": {"ok": cuda_ok, "output": cuda_out[:200] if cuda_ok else None,
                     "error": cuda_err, "time_s": round(cuda_time, 1)},
            "cpu":  {"ok": cpu_ok, "output": cpu_out[:200] if cpu_ok else None,
                     "error": cpu_err, "time_s": round(cpu_time, 1)},
        })

    # Write results
    out_path = "/home/IN.PGE.RJ.GOV.BR/fontesc/celeg/model_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results written to {out_path}")
    print(f"{'='*60}")

    # Summary
    cuda_ok_count = sum(1 for m in results["models"] if m["cuda"]["ok"])
    cpu_ok_count = sum(1 for m in results["models"] if m["cpu"]["ok"])
    print(f"\nSUMMARY:")
    print(f"  Total models tested: {len(results['models'])}")
    print(f"  CUDA OK: {cuda_ok_count}/{len(results['models'])}")
    print(f"  CPU  OK: {cpu_ok_count}/{len(results['models'])}")
    for m in results["models"]:
        cuda_s = "OK " if m["cuda"]["ok"] else "FAIL"
        cpu_s = "OK " if m["cpu"]["ok"] else "FAIL"
        print(f"  {cuda_s} {cpu_s}  {m['repo']}")

if __name__ == "__main__":
    main()
