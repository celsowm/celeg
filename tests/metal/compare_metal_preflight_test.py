#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "compare_metal", ROOT / "benchmarks" / "compare_metal.py")
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class CompareMetalPreflightTest(unittest.TestCase):
    @staticmethod
    def report(kernel=None):
        kernels = [] if kernel is None else [{"kernel": kernel, "count": 82}]
        return {
            "numerical_policy": "fast",
            "backend": "metal-native policy_requested=fast policy_effective=fast",
            "profile_dispatches": {
                "mode": "counts",
                "schedule_distorted": False,
                "prefill": {"kernels": kernels},
            },
        }

    def test_rejects_unavailable_fast_policy(self):
        report = self.report()
        report["backend"] = "metal-native policy_requested=fast policy_effective=strict"
        with mock.patch.object(COMPARE, "run_json", return_value=report):
            with self.assertRaisesRegex(RuntimeError, "fast policy is unavailable"):
                COMPARE.validate_fast_preflight(["celeg-metal-bench"], pathlib.Path("model.gguf"))

    def test_rejects_scalar_gemm(self):
        report = self.report("celeg_matmul_q4k")
        with mock.patch.object(COMPARE, "run_json", return_value=report):
            with self.assertRaisesRegex(RuntimeError, "scalar GEMM"):
                COMPARE.validate_fast_preflight(["celeg-metal-bench"], pathlib.Path("model.gguf"))

    def test_rejects_strict_tensor_fallback(self):
        report = self.report("celeg_matmul_tensor_q8_0")
        report["backend"] += " fast_q8_0=no"
        with mock.patch.object(COMPARE, "run_json", return_value=report):
            with self.assertRaisesRegex(RuntimeError, "fell back to Strict TensorOps"):
                COMPARE.validate_fast_preflight(
                    ["celeg-metal-bench"], pathlib.Path("model.gguf"))

    def test_accepts_tensor_gemm(self):
        report = self.report("celeg_matmul_tensor_q4k_relaxed")
        report.update({
            "build_commit": "abc",
            "build_dirty": False,
            "metal_source_sha256": "def",
        })
        with mock.patch.object(COMPARE, "run_json", return_value=report):
            result = COMPARE.validate_fast_preflight(
                ["celeg-metal-bench"], pathlib.Path("model.gguf"))
        self.assertEqual(result["build_commit"], "abc")
        self.assertEqual(
            result["dispatch_histogram"]["celeg_matmul_tensor_q4k_relaxed"], 82)

    def test_accepts_strict_precision_q6_fast_kernel(self):
        report = self.report("celeg_matmul_tensor_q6k_fast_strict")
        report["backend"] += (
            " fast_q6k=yes fast_q6k_precision=selective_m5_ffn_gate_0_7"
        )
        with mock.patch.object(COMPARE, "run_json", return_value=report):
            result = COMPARE.validate_fast_preflight(
                ["celeg-metal-bench"], pathlib.Path("LFM2.5-350M-Q6_K.gguf"))
        self.assertEqual(
            result["dispatch_histogram"]["celeg_matmul_tensor_q6k_fast_strict"], 82)

    def test_rejects_unavailable_selective_q6_family(self):
        report = self.report("celeg_matmul_tensor_q6k")
        report["backend"] += (
            " fast_q6k=no fast_q6k_precision=strict"
        )
        with mock.patch.object(COMPARE, "run_json", return_value=report):
            with self.assertRaisesRegex(RuntimeError, "fell back to Strict TensorOps"):
                COMPARE.validate_fast_preflight(
                    ["celeg-metal-bench"], pathlib.Path("LFM2.5-350M-Q6_K.gguf"))

    def test_llama_decode_uses_direct_generation_row(self):
        rows = [
            {"n_prompt": 512, "n_gen": 0, "samples_ns": [50_000_000]},
            {"n_prompt": 0, "n_gen": 8, "samples_ns": [20_000_000]},
            {"n_prompt": 512, "n_gen": 8, "samples_ns": [40_000_000]},
        ]
        result = COMPARE.llama_rows(rows, 512, 8)
        self.assertEqual(result["prefill_batched"]["median_milliseconds"], 50.0)
        self.assertEqual(result["decode"]["median_milliseconds"], 20.0)
        self.assertEqual(result["combined"]["median_milliseconds"], 40.0)


if __name__ == "__main__":
    unittest.main()
