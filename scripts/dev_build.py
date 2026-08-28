"""Build, test, and smoke orchestration."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Mapping, Sequence

from dev_environment import DevError, Environment, env_value, run_capture, which

ROOT = pathlib.Path(__file__).resolve().parent.parent
HF_REPO = "LiquidAI/LFM2.5-230M"


def print_visible(value: str, *, end: str = "\n") -> None:
    """Emit build output even when the Windows console is not UTF-8."""
    try:
        print(value, end=end)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((value + end).encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()


def build_directory(args: argparse.Namespace, environment: Environment) -> pathlib.Path:
    if args.build_dir:
        return pathlib.Path(args.build_dir).expanduser().resolve()
    flavor = f"{environment.platform_name}-{environment.backend}-{args.build_type.lower()}"
    return ROOT / "out" / flavor


def cmake_string_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_msvc_prefix_cache_script(directory: pathlib.Path, prefix: str) -> pathlib.Path:
    """Write an initial-cache script carrying the localized /showIncludes prefix.

    CMake mangles non-ASCII `-D` command-line argument values through the
    process's ANSI/OEM codepage before writing them into generated Ninja
    files, but preserves UTF-8 *file* content byte-for-byte. Ninja's own
    subprocess capture of cl.exe -- unlike a bare Python subprocess.PIPE --
    yields UTF-8 for the localized note, so `msvc_deps_prefix` in the
    generated rules.ninja must also be UTF-8 to byte-match it. Routing the
    value through a `-C` script instead of `-D` is what makes that happen.
    """
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "celeg_msvc_showincludes_prefix.cmake"
    script.write_text(
        "set(CELEG_MSVC_SHOWINCLUDES_PREFIX "
        f"{cmake_string_literal(prefix)} CACHE STRING \"\" FORCE)\n",
        encoding="utf-8",
    )
    return script


def configure_command(
    args: argparse.Namespace,
    environment: Environment,
    directory: pathlib.Path,
    *,
    allow_unsupported: bool = False,
) -> list[str]:
    assert environment.cmake
    command = [
        str(environment.cmake.path),
        "--fresh",
        "-S",
        str(ROOT),
        "-B",
        str(directory),
        "-G",
        "Ninja",
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DCELEG_ENABLE_CUDA={'ON' if environment.backend == 'cuda' else 'OFF'}",
        f"-DCELEG_ENABLE_METAL={'ON' if environment.backend == 'metal' else 'OFF'}",
        "-DCELEG_BUILD_TESTS=ON",
        f"-DCELEG_RUN_CUDA_TESTS={'ON' if args.celeg_tests == 'on' else 'OFF'}",
        f"-DCELEG_RUN_METAL_TESTS={'ON' if args.celeg_tests == 'on' else 'OFF'}",
    ]
    # Do not let CMake rediscover a newer preview toolset from the global
    # machine when the harness already selected a CUDA-compatible MSVC host.
    # This also makes --fresh deterministic after a prior build cached a
    # different compiler.
    if environment.platform_name == "windows" and environment.compiler:
        command.extend(
            [
                f"-DCMAKE_C_COMPILER={environment.compiler.path}",
                f"-DCMAKE_CXX_COMPILER={environment.compiler.path}",
            ]
        )
    if environment.backend == "cuda":
        assert environment.nvcc
        command.extend(
            [
                f"-DCMAKE_CUDA_COMPILER={environment.nvcc.path}",
                f"-DCMAKE_CUDA_ARCHITECTURES={environment.gpu_arch}",
                # Pin the host compiler by absolute path. NVCC's compiler
                # identification can otherwise invoke bare `cl.exe` from a
                # different process environment than CMake's CXX compiler.
                "-DCMAKE_CUDA_FLAGS_INIT=--use-local-env",
            ]
        )
        if allow_unsupported:
            command.append("-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler")
    if environment.msvc_include_prefix:
        prefix_script = write_msvc_prefix_cache_script(directory, environment.msvc_include_prefix)
        command.extend(["-C", str(prefix_script)])
    serve = env_value(environment.values, "CELEG_ENABLE_SERVE")
    if serve:
        command.append(f"-DCELEG_ENABLE_SERVE={serve}")
    if getattr(args, "asan", False):
        command.append("-DCELEG_ASAN=ON")
    return command


def run_visible(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: pathlib.Path = ROOT,
    capture: bool = False,
    filter_includes: bool = False,
) -> subprocess.CompletedProcess[str]:
    rendered_command = "+ " + subprocess.list2cmdline(list(command))
    # Visual Studio's show-includes prefix can contain Unicode characters on
    # localized Windows installations.  Logging a command must never prevent
    # the command itself from running just because the attached console is
    # still configured for cp1252.
    print_visible(rendered_command)
    if capture:
        result = run_capture(command, env=env, cwd=cwd)
        if result.stdout:
            print_visible(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        return result
    if filter_includes:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout
        for line in process.stdout:
            lower = line.casefold()
            if "including file:" in lower or "incluindo arquivo:" in lower:
                continue
            print_visible(line, end="")
        return subprocess.CompletedProcess(list(command), process.wait())
    return subprocess.run(list(command), cwd=str(cwd), env=dict(env), text=True, check=False)


def configure(args: argparse.Namespace, environment: Environment, directory: pathlib.Path) -> None:
    result = run_visible(
        configure_command(args, environment, directory),
        env=environment.values,
        capture=True,
    )
    if result.returncode == 0:
        return
    unsupported = re.search(
        r"unsupported\s+(?:Microsoft\s+)?Visual Studio|allow-unsupported-compiler",
        result.stdout or "",
        re.IGNORECASE,
    )
    if environment.backend == "cuda" and unsupported:
        print_visible("configure: NVCC rejected this host compiler; retrying with -allow-unsupported-compiler")
        retry = run_visible(
            configure_command(args, environment, directory, allow_unsupported=True),
            env=environment.values,
            capture=True,
        )
        if retry.returncode == 0:
            return
    raise DevError("CMake configure failed.")


class BuildCoordinator:
    """Owns CMake configuration and compilation for one build directory."""

    def __init__(self, args: argparse.Namespace, environment: Environment,
                 directory: pathlib.Path) -> None:
        self.args = args
        self.environment = environment
        self.directory = directory

    def run(self) -> None:
        configure(self.args, self.environment, self.directory)
        assert self.environment.cmake
        result = run_visible(
            [
                str(self.environment.cmake.path),
                "--build",
                str(self.directory),
                "--parallel",
                str(self.args.jobs),
            ],
            env=self.environment.values,
            filter_includes=self.environment.platform_name == "windows",
        )
        if result.returncode != 0:
            raise DevError("Build failed.")


class TestCoordinator:
    """Owns language-level and CTest validation for a completed build."""

    def __init__(self, args: argparse.Namespace, environment: Environment,
                 directory: pathlib.Path) -> None:
        self.args = args
        self.environment = environment
        self.directory = directory

    def run_python(self) -> None:
        result = run_visible(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "dev_cli_test.py",
            ],
            env=self.environment.values,
        )
        if result.returncode != 0:
            raise DevError("Developer CLI unit tests failed.")

    def run_ctest(self) -> None:
        assert self.environment.cmake
        ctest_name = "ctest.exe" if self.environment.platform_name == "windows" else "ctest"
        ctest = self.environment.cmake.path.with_name(ctest_name)
        if not ctest.is_file():
            found = which(ctest_name, self.environment.values)
            if not found:
                raise DevError("CTest was not found.")
            ctest = found
        result = run_visible(
            [
                str(ctest),
                "--test-dir",
                str(self.directory),
                "--output-on-failure",
                "--parallel",
                str(self.args.jobs),
            ],
            env=self.environment.values,
        )
        if result.returncode != 0:
            raise DevError("CTest failed.")

    def run(self) -> None:
        self.run_python()
        self.run_ctest()


def executable(directory: pathlib.Path, name: str, environment: Environment) -> pathlib.Path:
    suffix = ".exe" if environment.platform_name == "windows" else ""
    candidate = directory / f"{name}{suffix}"
    if candidate.is_file():
        return candidate
    raise DevError(f"Expected executable was not built: {name}{suffix}")


class SmokeCoordinator:
    """Owns executable smoke checks and optional cached-model inference."""

    def __init__(self, args: argparse.Namespace, environment: Environment,
                 directory: pathlib.Path) -> None:
        self.args = args
        self.environment = environment
        self.directory = directory

    def _runner(self) -> pathlib.Path:
        if self.environment.backend == "cuda":
            name = "celeg-run"
        elif self.environment.backend == "metal":
            name = "celeg-metal-run"
        else:
            name = "celeg-cpu-run"
        return executable(self.directory, name, self.environment)

    def _commands(self, runner: pathlib.Path) -> list[list[str]]:
        commands = [
            [str(runner), "--help"],
            [str(executable(self.directory, "cpu_c_api_smoke_test", self.environment))],
        ]
        if self.environment.backend == "cuda" and self.args.celeg_tests == "on":
            commands.append([
                str(executable(self.directory, "expert_residency_test", self.environment))
            ])
        if self.environment.backend == "metal" and self.args.celeg_tests == "on":
            commands.append([
                str(executable(self.directory, "metal_device_test", self.environment))
            ])
            commands.append([
                str(executable(self.directory, "metal_recurrent_test", self.environment))
            ])
            if self.environment.quantized_checkpoint:
                commands.append([
                    str(executable(self.directory, "metal_quantization_test", self.environment)),
                    str(self.environment.quantized_checkpoint),
                ])
            if self.environment.checkpoint:
                commands.append([
                    str(executable(self.directory, "metal_inference_test", self.environment)),
                    str(self.environment.checkpoint),
                ])
                commands.append([
                    str(executable(self.directory, "metal_c_api_smoke_test", self.environment)),
                    str(self.environment.checkpoint),
                ])
                commands.append([
                    str(executable(self.directory, "metal_service_test", self.environment)),
                    str(self.environment.checkpoint),
                ])
        return commands

    def _run_commands(self, commands: Sequence[Sequence[str]]) -> None:
        for command in commands:
            result = run_visible(command, env=self.environment.values)
            if result.returncode != 0:
                raise DevError(f"Smoke command failed: {pathlib.Path(command[0]).name}")

    def _run_cached_inference(self, runner: pathlib.Path) -> None:
        if self.environment.backend == "cpu":
            print_visible("smoke: model inference skipped for CPU verification")
            return
        if self.environment.backend == "cuda" and not self.environment.gpu_name:
            print_visible("smoke: no CUDA device is visible; model inference skipped")
            return
        if self.environment.checkpoint:
            if self.environment.backend == "metal":
                model_argument = ["--model", str(self.environment.checkpoint)]
            else:
                model_argument = ["--repo", HF_REPO]
            result = run_visible(
                [
                    str(runner),
                    *model_argument,
                    "--prompt",
                    "Hello",
                    "--max-new-tokens",
                    "4",
                ],
                env=self.environment.values,
            )
            if result.returncode != 0:
                raise DevError("Cached model inference smoke test failed.")
        else:
            print_visible(f"smoke: {HF_REPO} is not cached; model inference skipped")
        if self.environment.backend == "metal" and self.environment.quantized_checkpoint:
            result = run_visible(
                [
                    str(runner),
                    "--model",
                    str(self.environment.quantized_checkpoint),
                    "--prompt",
                    "Hello",
                    "--max-new-tokens",
                    "4",
                ],
                env=self.environment.values,
            )
            if result.returncode != 0:
                raise DevError("Cached GGUF Metal inference smoke test failed.")
            result = run_visible(
                [
                    str(executable(self.directory, "metal_c_api_smoke_test", self.environment)),
                    str(self.environment.quantized_checkpoint),
                ],
                env=self.environment.values,
            )
            if result.returncode != 0:
                raise DevError("Cached GGUF Metal C API smoke test failed.")
        if self.environment.backend == "metal" and self.environment.moe_checkpoint:
            result = run_visible(
                [
                    str(runner),
                    "--model",
                    str(self.environment.moe_checkpoint),
                    "--context",
                    "64",
                    "--prompt",
                    "Hello",
                    "--max-new-tokens",
                    "1",
                ],
                env=self.environment.values,
            )
            if result.returncode != 0:
                raise DevError("Cached Metal MoE inference smoke test failed.")

    def run(self) -> None:
        runner = self._runner()
        self._run_commands(self._commands(runner))
        self._run_cached_inference(runner)
