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
    def test_rejects_unavailable_fast_policy(self):
        report = {
            "numerical_policy": "fast",
            "backend": "metal-native policy_requested=fast policy_effective=strict",
        }
        with mock.patch.object(COMPARE, "run_json", return_value=(report, "")):
            with self.assertRaisesRegex(RuntimeError, "fast policy is unavailable"):
                COMPARE.validate_fast_preflight(["celeg-metal-bench"], pathlib.Path("model.gguf"))

    def test_rejects_scalar_gemm(self):
        report = {
            "numerical_policy": "fast",
            "backend": "metal-native policy_requested=fast policy_effective=fast",
        }
        stderr = "metal dispatch profile\n  celeg_matmul_q4k=82\n"
        with mock.patch.object(COMPARE, "run_json", return_value=(report, stderr)):
            with self.assertRaisesRegex(RuntimeError, "scalar GEMM"):
                COMPARE.validate_fast_preflight(["celeg-metal-bench"], pathlib.Path("model.gguf"))

    def test_rejects_strict_tensor_fallback(self):
        report = {
            "numerical_policy": "fast",
            "backend": "metal-native policy_requested=fast policy_effective=fast fast_q8_0=no",
        }
        stderr = "metal dispatch profile\n  celeg_matmul_tensor_q8_0=82\n"
        with mock.patch.object(COMPARE, "run_json", return_value=(report, stderr)):
            with self.assertRaisesRegex(RuntimeError, "fell back to Strict TensorOps"):
                COMPARE.validate_fast_preflight(
                    ["celeg-metal-bench"], pathlib.Path("model.gguf"))

    def test_accepts_tensor_gemm(self):
        report = {
            "numerical_policy": "fast",
            "backend": "metal-native policy_requested=fast policy_effective=fast",
            "build_commit": "abc",
            "build_dirty": False,
            "metal_source_sha256": "def",
        }
        stderr = "metal dispatch profile\n  celeg_matmul_tensor_q4k_relaxed=82\n"
        with mock.patch.object(COMPARE, "run_json", return_value=(report, stderr)):
            result = COMPARE.validate_fast_preflight(
                ["celeg-metal-bench"], pathlib.Path("model.gguf"))
        self.assertEqual(result["build_commit"], "abc")
        self.assertEqual(
            result["dispatch_histogram"]["celeg_matmul_tensor_q4k_relaxed"], 82)


if __name__ == "__main__":
    unittest.main()
