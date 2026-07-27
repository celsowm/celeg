from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("lfm_dev", ROOT / "scripts" / "dev.py")
assert SPEC and SPEC.loader
dev = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dev
SPEC.loader.exec_module(dev)


def completed(output: str = "", code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, output)


class DevCliTest(unittest.TestCase):
    def test_version_key_orders_toolkits_numerically(self) -> None:
        paths = ["CUDA/v9.2/bin/nvcc", "CUDA/v13.1/bin/nvcc", "CUDA/v12.8/bin/nvcc"]
        self.assertEqual(sorted(paths, key=dev.version_key, reverse=True)[0], paths[1])

    def test_normalize_architecture(self) -> None:
        self.assertEqual(dev.normalize_arch("sm_86"), "86")
        self.assertEqual(dev.normalize_arch("12.0"), "120")
        self.assertEqual(dev.normalize_arch("native"), "native")
        with self.assertRaises(Exception):
            dev.normalize_arch("latest")

    def test_runtime_layout_supports_cuda_13_x64(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "bin" / "x64").mkdir(parents=True)
            (root / "bin" / "x64" / "cublas64_13.dll").touch()
            (root / "bin" / "x64" / "cublasLt64_13.dll").touch()
            directories = dev.cuda_runtime_dirs(root)
            dlls = dev.cuda_runtime_dlls(directories)
            self.assertEqual(directories[0], (root / "bin" / "x64").resolve())
            self.assertEqual({path.name for path in dlls}, {"cublas64_13.dll", "cublasLt64_13.dll"})

    def test_select_cuda_rejects_unsupported_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            old = pathlib.Path(temporary) / "v12.4" / "bin" / "nvcc"
            new = pathlib.Path(temporary) / "v13.1" / "bin" / "nvcc"
            old.parent.mkdir(parents=True)
            new.parent.mkdir(parents=True)
            old.touch()
            new.touch()

            def runner(command, **_kwargs):
                path = str(command[0])
                if command[1] == "--version":
                    return completed("Cuda compilation tools, release " + ("12.4" if "12.4" in path else "13.1"))
                return completed("compute_80\ncompute_86\n" + ("compute_120\n" if "13.1" in path else ""))

            selected = dev.select_cuda([old, new], "120", runner)
            self.assertIsNotNone(selected)
            self.assertEqual(selected.path, new)

    def test_checkpoint_lookup_uses_hf_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = (
                pathlib.Path(temporary)
                / "hub"
                / "models--LiquidAI--LFM2.5-230M"
                / "snapshots"
                / "commit"
            )
            snapshot.mkdir(parents=True)
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (snapshot / name).touch()
            self.assertEqual(dev.find_checkpoint({"HF_HOME": temporary}), snapshot.resolve())

    def test_discover_vcvars_uses_vswhere_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            vswhere = root / "vswhere.exe"
            vcvars = root / "VS" / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            vswhere.touch()
            vcvars.parent.mkdir(parents=True)
            vcvars.touch()
            env = {"PATH": str(root)}

            with mock.patch.object(dev, "find_vswhere", return_value=vswhere):
                found = dev.discover_vcvars(env, lambda *_args, **_kwargs: completed(str(root / "VS")))
            self.assertEqual(found, vcvars.resolve())

    def test_detect_msvc_include_prefix_is_language_independent(self) -> None:
        output = "Observação: incluindo arquivo: C:\\SDK\\include\\stddef.h\r\n"
        prefix = dev.detect_msvc_include_prefix(
            pathlib.Path("cl.exe"),
            {},
            lambda *_args, **_kwargs: completed(output),
        )
        self.assertEqual(prefix, "Observação: incluindo arquivo: ")

    def test_configure_command_uses_discovered_windows_cuda(self) -> None:
        environment = dev.Environment(
            values={},
            platform_name="windows",
            cmake=dev.Tool(pathlib.Path("cmake.exe")),
            ninja=dev.Tool(pathlib.Path("ninja.exe")),
            compiler=dev.Tool(pathlib.Path("cl.exe")),
            msvc_include_prefix="Note: including file: ",
            vcvars=None,
            nvcc=dev.Tool(pathlib.Path("nvcc.exe")),
            toolkit_root=None,
            runtime_dirs=[],
            runtime_dlls=[],
            gpu_name="GPU",
            driver_version="1",
            gpu_arch="86",
            requested_backend="cuda",
            backend="cuda",
            checkpoint=None,
            errors=[],
            warnings=[],
        )
        args = mock.Mock(build_type="RelWithDebInfo", runtime_tests="on")
        command = dev.configure_command(args, environment, pathlib.Path("out"))
        self.assertFalse(any(value.startswith("-DCMAKE_CUDA_HOST_COMPILER=") for value in command))
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", command)
        self.assertIn("-DLFM_MSVC_SHOWINCLUDES_PREFIX=Note: including file: ", command)
        self.assertNotIn("-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler", command)


if __name__ == "__main__":
    unittest.main()
