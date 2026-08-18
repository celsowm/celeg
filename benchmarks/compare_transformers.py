#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TOKEN_IDS_PREFIX = "CELEG_GENERATED_TOKEN_IDS="


class ComparisonError(RuntimeError):
    pass


def run_checked(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise ComparisonError(
            f"command exited with {proc.returncode}: {' '.join(cmd)}\n{proc.stdout}"
        )
    return proc.stdout


def parse_input_tokens(stdout: str) -> list[int]:
    candidates = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped and all(part.lstrip("-").isdigit() for part in stripped.split()):
            candidates.append([int(part) for part in stripped.split()])
    if len(candidates) != 1:
        raise ComparisonError(
            f"expected exactly one CELEG input-token line, found {len(candidates)}"
        )
    return candidates[0]


def parse_generated_tokens(stdout: str) -> list[int]:
    records = [
        line[len(TOKEN_IDS_PREFIX):].strip()
        for line in stdout.splitlines()
        if line.startswith(TOKEN_IDS_PREFIX)
    ]
    if len(records) != 1:
        raise ComparisonError(
            f"expected exactly one generated-token record, found {len(records)}"
        )
    try:
        value = json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise ComparisonError("invalid CELEG generated-token JSON") from exc
    if not isinstance(value, list) or any(type(token) is not int for token in value):
        raise ComparisonError("CELEG generated-token record is not an integer array")
    return value


def first_mismatch(
    actual: list[int], expected: list[int]
) -> tuple[int, int | None, int | None] | None:
    for index, (lhs, rhs) in enumerate(zip(actual, expected)):
        if lhs != rhs:
            return index, lhs, rhs
    if len(actual) != len(expected):
        index = min(len(actual), len(expected))
        lhs = actual[index] if index < len(actual) else None
        rhs = expected[index] if index < len(expected) else None
        return index, lhs, rhs
    return None


def celeg_base_command(args: argparse.Namespace) -> list[str]:
    model_arg = "--model" if Path(args.model).exists() else "--repo"
    return [
        str(args.celeg_run),
        model_arg, args.model,
        "--prompt", args.prompt,
        "--raw",
        "--context", str(args.context),
        "--weight-mode", args.weight_mode,
        "--kv-cache", args.kv_cache,
        "--top-k", "1",
        "--top-p", "1",
        "--temperature", "1",
        "--repetition-penalty", "1",
        "--seed", "1",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CELEG greedy generation against Transformers token-for-token."
    )
    parser.add_argument(
        "--model", required=True,
        help="Hugging Face repository id or local checkpoint directory",
    )
    parser.add_argument("--celeg-run", required=True, type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--weight-mode", default="bf16")
    parser.add_argument("--kv-cache", default="bf16")
    parser.add_argument(
        "--device", default=None,
        help="Torch device; defaults to cuda when available, otherwise cpu",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-logits", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ComparisonError("--max-new-tokens must be positive")
    if not args.celeg_run.is_file():
        raise ComparisonError(f"celeg-run not found: {args.celeg_run}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ComparisonError(
            "Transformers oracle unavailable: install torch and transformers; "
            "this comparison must not be treated as passed"
        ) from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype="auto",
    )
    model.eval()
    model.to(device)

    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    reference_input = encoded["input_ids"][0].tolist()

    celeg_input_stdout = run_checked(celeg_base_command(args) + ["--tokens-only"])
    celeg_input = parse_input_tokens(celeg_input_stdout)
    mismatch = first_mismatch(celeg_input, reference_input)
    if mismatch is not None:
        index, actual, expected = mismatch
        raise ComparisonError(
            "input-token mismatch before inference at index "
            f"{index}: celeg={actual}, transformers={expected}; "
            f"celeg_len={len(celeg_input)}, transformers_len={len(reference_input)}"
        )

    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        reference_forward = model(**encoded)
        reference_logits = reference_forward.logits[0, -1].float().cpu()
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    reference_tokens = generated[0, encoded["input_ids"].shape[1]:].tolist()

    if not args.skip_logits:
        with tempfile.TemporaryDirectory(prefix="celeg-oracle-") as tmp:
            logits_path = Path(tmp) / "celeg_logits.f32"
            run_checked(
                celeg_base_command(args)
                + ["--max-new-tokens", "0", "--dump-logits", str(logits_path)]
            )
            byte_count = logits_path.stat().st_size
            if byte_count % 4 != 0:
                raise ComparisonError(
                    f"CELEG logits file has invalid byte size: {byte_count}"
                )
            celeg_logits = torch.from_file(
                str(logits_path), dtype=torch.float32, size=byte_count // 4
            ).clone()
            if celeg_logits.numel() != reference_logits.numel():
                raise ComparisonError(
                    "vocabulary/logit-size mismatch: "
                    f"celeg={celeg_logits.numel()}, transformers={reference_logits.numel()}"
                )
            celeg_top1 = int(torch.argmax(celeg_logits).item())
            reference_top1 = int(torch.argmax(reference_logits).item())
            abs_diff = (celeg_logits - reference_logits).abs()
            print(
                "logits: "
                f"celeg_top1={celeg_top1} transformers_top1={reference_top1} "
                f"max_abs={float(abs_diff.max()):.6g} "
                f"mean_abs={float(abs_diff.mean()):.6g}"
            )
            if celeg_top1 != reference_top1:
                raise ComparisonError(
                    "first-token argmax mismatch: "
                    f"celeg={celeg_top1}, transformers={reference_top1}"
                )

    celeg_stdout = run_checked(
        celeg_base_command(args)
        + ["--max-new-tokens", str(args.max_new_tokens), "--runtime-tokens"]
    )
    celeg_tokens = parse_generated_tokens(celeg_stdout)
    mismatch = first_mismatch(celeg_tokens, reference_tokens)
    if mismatch is not None:
        index, actual, expected = mismatch
        raise ComparisonError(
            f"generation mismatch at token {index}: "
            f"celeg={actual}, transformers={expected}; "
            f"celeg_len={len(celeg_tokens)}, transformers_len={len(reference_tokens)}\n"
            f"celeg_tokens={celeg_tokens}\n"
            f"transformers_tokens={reference_tokens}"
        )

    print(
        f"PASS input_tokens={len(reference_input)} "
        f"generated_tokens={len(reference_tokens)}"
    )
    print(f"tokens={reference_tokens}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ComparisonError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
