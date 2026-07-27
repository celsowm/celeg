#!/usr/bin/env python3
"""Portable build, test, smoke, and environment diagnostics for lfm25."""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Iterable, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parent.parent
HF_REPO = "LiquidAI/LFM2.5-230M"
Runner = Callable[..., subprocess.CompletedProcess[str]]


class DevError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Tool:
    path: pathlib.Path
    version: str = ""


@dataclasses.dataclass
class Environment:
    values: dict[str, str]
    platform_name: str
    cmake: Tool | None
    ninja: Tool | None
    compiler: Tool | None
    msvc_include_prefix: str
    vcvars: pathlib.Path | None
    nvcc: Tool | None
    toolkit_root: pathlib.Path | None
    runtime_dirs: list[pathlib.Path]
    runtime_dlls: list[pathlib.Path]
    gpu_name: str
    driver_version: str
    gpu_arch: str
    requested_backend: str
    backend: str
    checkpoint: pathlib.Path | None
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def run_capture(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def version_key(value: str | pathlib.Path) -> tuple[int, ...]:
    matches = re.findall(r"\d+", str(value))
    return tuple(int(part) for part in matches[-3:]) if matches else ()


def executable_version(path: pathlib.Path, runner: Runner = run_capture) -> str:
    result = runner([str(path), "--version"])
    text = result.stdout or ""
    cuda = re.search(r"release\s+(\d+(?:\.\d+)*)", text, re.IGNORECASE)
    if cuda:
        return cuda.group(1)
    generic = re.search(r"(\d+\.\d+(?:\.\d+)*)", text)
    return generic.group(1) if generic else ""


def which(name: str, env: Mapping[str, str]) -> pathlib.Path | None:
    found = shutil.which(name, path=env_value(env, "PATH"))
    return pathlib.Path(found).resolve() if found else None


def env_value(env: Mapping[str, str], name: str) -> str | None:
    if name in env:
        return env[name]
    if os.name == "nt":
        expected = name.casefold()
        for key, value in env.items():
            if key.casefold() == expected:
                return value
    return None


def prepend_path(env: dict[str, str], paths: Iterable[pathlib.Path]) -> None:
    additions = [str(path) for path in paths if path.is_dir()]
    if additions:
        env["PATH"] = os.pathsep.join(additions + [env_value(env, "PATH") or ""])


def find_vswhere(env: Mapping[str, str]) -> pathlib.Path | None:
    explicit = which("vswhere.exe", env)
    if explicit:
        return explicit
    base = env_value(env, "ProgramFiles(x86)") or env_value(env, "ProgramFiles")
    if not base:
        return None
    candidate = pathlib.Path(base) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    return candidate if candidate.is_file() else None


def discover_vcvars(
    env: Mapping[str, str],
    runner: Runner = run_capture,
) -> pathlib.Path | None:
    override = env_value(env, "VCVARSALL")
    if override and pathlib.Path(override).is_file():
        return pathlib.Path(override).resolve()

    for variable in ("VSINSTALLDIR", "VCINSTALLDIR"):
        value = env_value(env, variable)
        if not value:
            continue
        base = pathlib.Path(value)
        if variable == "VCINSTALLDIR":
            candidate = base / "Auxiliary" / "Build" / "vcvars64.bat"
        else:
            candidate = base / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if candidate.is_file():
            return candidate.resolve()

    vswhere = find_vswhere(env)
    if not vswhere:
        return None
    result = runner(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        env=env,
    )
    if result.returncode != 0:
        return None
    installations = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not installations:
        return None
    candidate = (
        pathlib.Path(installations[-1])
        / "VC"
        / "Auxiliary"
        / "Build"
        / "vcvars64.bat"
    )
    return candidate.resolve() if candidate.is_file() else None


def capture_vcvars(
    vcvars: pathlib.Path,
    env: Mapping[str, str],
    runner: Runner = run_capture,
) -> dict[str, str]:
    script_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".bat",
            encoding="utf-8",
            newline="\r\n",
            delete=False,
        ) as script:
            script.write("@echo off\n")
            script.write(f'call "{vcvars}" >nul\n')
            script.write("if errorlevel 1 exit /b %errorlevel%\n")
            script.write("set\n")
            script_path = pathlib.Path(script.name)
        result = runner(["cmd.exe", "/d", "/c", str(script_path)], env=env)
    finally:
        if script_path:
            script_path.unlink(missing_ok=True)
    if result.returncode != 0:
        detail = next((line.strip() for line in reversed(result.stdout.splitlines()) if line.strip()), "")
        suffix = f": {detail}" if detail else ""
        raise DevError(f"vcvars64.bat failed with exit code {result.returncode}{suffix}")
    captured = dict(env)
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            captured[key] = value
    return captured


def detect_msvc_include_prefix(
    compiler: pathlib.Path,
    env: Mapping[str, str],
    runner: Runner = run_capture,
) -> str:
    with tempfile.TemporaryDirectory() as temporary:
        directory = pathlib.Path(temporary)
        source = directory / "showincludes.cpp"
        source.write_text("#include <stddef.h>\n", encoding="ascii")
        result = runner(
            [
                str(compiler),
                "/nologo",
                "/showIncludes",
                "/c",
                str(source),
                f"/Fo{directory / 'showincludes.obj'}",
            ],
            env=env,
        )
    for line in result.stdout.splitlines():
        match = re.match(r"^(.*?)([A-Za-z]:[\\/].*stddef\.h)\s*$", line, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def cuda_candidates(env: Mapping[str, str], platform_name: str) -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    for variable in ("CMAKE_CUDA_COMPILER", "CUDACXX"):
        value = env_value(env, variable)
        if value:
            candidates.append(pathlib.Path(value))

    cuda_path = env_value(env, "CUDA_PATH")
    if cuda_path:
        candidates.append(pathlib.Path(cuda_path) / "bin" / ("nvcc.exe" if platform_name == "windows" else "nvcc"))

    on_path = which("nvcc.exe" if platform_name == "windows" else "nvcc", env)
    if on_path:
        candidates.append(on_path)

    if platform_name == "windows":
        base = env_value(env, "ProgramFiles")
        if base:
            pattern = pathlib.Path(base) / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v*" / "bin" / "nvcc.exe"
            candidates.extend(pathlib.Path(path) for path in glob.glob(str(pattern)))
    else:
        for pattern in ("/usr/local/cuda*/bin/nvcc", "/opt/cuda*/bin/nvcc"):
            candidates.extend(pathlib.Path(path) for path in glob.glob(pattern))

    unique: dict[str, pathlib.Path] = {}
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            unique[os.path.normcase(str(resolved))] = resolved
    return sorted(unique.values(), key=lambda path: version_key(path), reverse=True)


def nvcc_architectures(nvcc: pathlib.Path, runner: Runner = run_capture) -> set[str]:
    result = runner([str(nvcc), "--list-gpu-arch"])
    if result.returncode != 0:
        return set()
    return {
        match.group(1)
        for match in re.finditer(r"(?:compute_|sm_)?(\d{2,3}[a-z]?)", result.stdout)
    }


def select_cuda(
    candidates: Sequence[pathlib.Path],
    arch: str,
    runner: Runner = run_capture,
) -> Tool | None:
    probes: list[Tool] = []
    for candidate in candidates:
        version = executable_version(candidate, runner)
        supported = nvcc_architectures(candidate, runner)
        if arch != "native" and supported and arch not in supported:
            continue
        probes.append(Tool(candidate, version))
    if not probes:
        return None
    return max(probes, key=lambda tool: (version_key(tool.version), version_key(tool.path)))


def toolkit_root(nvcc: pathlib.Path) -> pathlib.Path:
    return nvcc.resolve().parent.parent


def cuda_runtime_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    candidates = [root / "bin" / "x64", root / "bin"]
    return [candidate.resolve() for candidate in candidates if candidate.is_dir()]


def cuda_runtime_dlls(runtime_dirs: Sequence[pathlib.Path]) -> list[pathlib.Path]:
    names = ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")
    found: dict[str, pathlib.Path] = {}
    for directory in runtime_dirs:
        for name in names:
            for path in directory.glob(name):
                found[path.name.lower()] = path.resolve()
    return sorted(found.values(), key=lambda path: path.name.lower())


def query_gpu(
    env: Mapping[str, str],
    platform_name: str | None = None,
    runner: Runner = run_capture,
) -> tuple[str, str, str]:
    system = platform_name or ("windows" if os.name == "nt" else platform.system().lower())
    smi = which("nvidia-smi.exe" if system == "windows" else "nvidia-smi", env)
    if not smi:
        return "", "", ""
    result = runner(
        [
            str(smi),
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        env=env,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "", "", ""
    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
    if len(fields) < 3:
        return "", "", ""
    arch = re.sub(r"[^0-9a-z]", "", fields[2].lower())
    return fields[0], fields[1], arch


def hf_cache_root(env: Mapping[str, str]) -> pathlib.Path:
    hub_cache = env_value(env, "HF_HUB_CACHE")
    hf_home = env_value(env, "HF_HOME")
    if hub_cache:
        return pathlib.Path(hub_cache).expanduser()
    if hf_home:
        return pathlib.Path(hf_home).expanduser() / "hub"
    return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def find_checkpoint(
    env: Mapping[str, str],
    repo: str = HF_REPO,
) -> pathlib.Path | None:
    owner, name = repo.split("/", 1)
    snapshots = hf_cache_root(env) / f"models--{owner}--{name}" / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        config = candidate / "config.json"
        tokenizer = candidate / "tokenizer.json"
        weights = list(candidate.glob("*.safetensors"))
        if config.is_file() and tokenizer.is_file() and weights:
            return candidate.resolve()
    return None


def discover_environment(
    requested_backend: str,
    requested_arch: str,
    *,
    source_env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    runner: Runner = run_capture,
) -> Environment:
    values = dict(source_env or os.environ)
    system = platform_name or ("windows" if os.name == "nt" else platform.system().lower())
    errors: list[str] = []
    warnings: list[str] = []
    vcvars: pathlib.Path | None = None

    msvc_include_prefix = ""
    if system == "windows":
        vcvars = discover_vcvars(values, runner)
        if vcvars:
            try:
                values = capture_vcvars(vcvars, values, runner)
                values["VSLANG"] = "1033"
            except DevError as error:
                errors.append(str(error))
        else:
            errors.append("Visual Studio C++ Build Tools were not found through the environment or vswhere.")

    cmake_path = which("cmake.exe" if system == "windows" else "cmake", values)
    ninja_path = which("ninja.exe" if system == "windows" else "ninja", values)
    compiler_name = "cl.exe" if system == "windows" else (env_value(values, "CXX") or "c++")
    compiler_path = which(compiler_name, values)

    cmake = Tool(cmake_path, executable_version(cmake_path, runner)) if cmake_path else None
    ninja = Tool(ninja_path, executable_version(ninja_path, runner)) if ninja_path else None
    compiler = Tool(compiler_path, executable_version(compiler_path, runner)) if compiler_path else None

    if not cmake:
        errors.append("CMake was not found on PATH.")
    if not ninja:
        errors.append("Ninja was not found on PATH.")
    if not compiler:
        errors.append("A C++ compiler was not found after environment initialization.")
    elif system == "windows":
        msvc_include_prefix = detect_msvc_include_prefix(compiler.path, values, runner)
        if not msvc_include_prefix:
            warnings.append("MSVC /showIncludes prefix could not be detected; Ninja output may be noisy.")

    gpu_name, driver_version, detected_arch = query_gpu(values, system, runner)
    arch = detected_arch if requested_arch == "native" and detected_arch else requested_arch
    candidates = cuda_candidates(values, system)
    nvcc = select_cuda(candidates, arch, runner) if candidates else None
    root = toolkit_root(nvcc.path) if nvcc else None
    runtime_dirs = cuda_runtime_dirs(root) if root else []
    runtime_dlls = cuda_runtime_dlls(runtime_dirs) if system == "windows" else []
    cuda_buildable = bool(nvcc and arch != "native")

    backend = requested_backend
    if backend == "auto":
        backend = "cuda" if cuda_buildable else "cpu"
        if candidates and not cuda_buildable:
            warnings.append("CUDA was found but no compatible concrete GPU architecture was detected; using CPU.")
    elif backend == "cuda":
        if not nvcc:
            errors.append("No CUDA toolkit supports the requested architecture.")
        if arch == "native":
            errors.append("CUDA architecture is native but no GPU compute capability was detected; pass --arch N.")

    if backend == "cuda" and system == "windows":
        required = ("cublas64_", "cublaslt64_")
        lower_names = [path.name.lower() for path in runtime_dlls]
        missing = [prefix for prefix in required if not any(name.startswith(prefix) for name in lower_names)]
        if missing:
            errors.append("Required CUDA runtime DLLs were not found under the selected toolkit.")

    if runtime_dirs:
        prepend_path(values, runtime_dirs)
    if nvcc:
        prepend_path(values, [nvcc.path.parent])

    return Environment(
        values=values,
        platform_name=system,
        cmake=cmake,
        ninja=ninja,
        compiler=compiler,
        msvc_include_prefix=msvc_include_prefix,
        vcvars=vcvars,
        nvcc=nvcc,
        toolkit_root=root,
        runtime_dirs=runtime_dirs,
        runtime_dlls=runtime_dlls,
        gpu_name=gpu_name,
        driver_version=driver_version,
        gpu_arch=arch,
        requested_backend=requested_backend,
        backend=backend,
        checkpoint=find_checkpoint(values),
        errors=errors,
        warnings=warnings,
    )


def tool_json(tool: Tool | None) -> dict[str, str] | None:
    return {"path": str(tool.path), "version": tool.version} if tool else None


def environment_json(environment: Environment) -> dict[str, object]:
    return {
        "ok": environment.ok,
        "platform": environment.platform_name,
        "requested_backend": environment.requested_backend,
        "backend": environment.backend,
        "cmake": tool_json(environment.cmake),
        "ninja": tool_json(environment.ninja),
        "compiler": tool_json(environment.compiler),
        "msvc_include_prefix": environment.msvc_include_prefix or None,
        "vcvars": str(environment.vcvars) if environment.vcvars else None,
        "cuda": {
            "nvcc": tool_json(environment.nvcc),
            "toolkit_root": str(environment.toolkit_root) if environment.toolkit_root else None,
            "runtime_dirs": [str(path) for path in environment.runtime_dirs],
            "runtime_dlls": [str(path) for path in environment.runtime_dlls],
        },
        "gpu": {
            "name": environment.gpu_name or None,
            "driver": environment.driver_version or None,
            "architecture": environment.gpu_arch,
        },
        "checkpoint": str(environment.checkpoint) if environment.checkpoint else None,
        "warnings": environment.warnings,
        "errors": environment.errors,
    }


def print_doctor(environment: Environment, as_json: bool) -> None:
    report = environment_json(environment)
    if as_json:
        print(json.dumps(report, indent=2))
        return

    def display(tool: Tool | None) -> str:
        return f"{tool.path} ({tool.version or 'unknown version'})" if tool else "<not found>"

    print(f"doctor: platform       {environment.platform_name}")
    print(f"doctor: backend        {environment.backend} (requested {environment.requested_backend})")
    print(f"doctor: cmake          {display(environment.cmake)}")
    print(f"doctor: ninja          {display(environment.ninja)}")
    print(f"doctor: compiler       {display(environment.compiler)}")
    print(f"doctor: nvcc           {display(environment.nvcc)}")
    print(f"doctor: gpu            {environment.gpu_name or '<not detected>'}")
    print(f"doctor: driver         {environment.driver_version or '<not detected>'}")
    print(f"doctor: architecture   {environment.gpu_arch}")
    print(f"doctor: checkpoint     {environment.checkpoint or '<not cached>'}")
    for warning in environment.warnings:
        print(f"doctor: warning: {warning}")
    for error in environment.errors:
        print(f"doctor: error: {error}")
    print(f"doctor: RESULT: {'ready' if environment.ok else 'not ready'}")


def build_directory(args: argparse.Namespace, environment: Environment) -> pathlib.Path:
    if args.build_dir:
        return pathlib.Path(args.build_dir).expanduser().resolve()
    flavor = f"{environment.platform_name}-{environment.backend}-{args.build_type.lower()}"
    return ROOT / "out" / flavor


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
        f"-DLFM_ENABLE_CUDA={'ON' if environment.backend == 'cuda' else 'OFF'}",
        "-DLFM_BUILD_TESTS=ON",
        f"-DLFM_RUN_CUDA_TESTS={'ON' if args.runtime_tests == 'on' else 'OFF'}",
    ]
    if environment.backend == "cuda":
        assert environment.nvcc
        command.extend(
            [
                f"-DCMAKE_CUDA_COMPILER={environment.nvcc.path}",
                f"-DCMAKE_CUDA_ARCHITECTURES={environment.gpu_arch}",
            ]
        )
        if allow_unsupported:
            command.append("-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler")
    if environment.msvc_include_prefix:
        command.append(f"-DLFM_MSVC_SHOWINCLUDES_PREFIX={environment.msvc_include_prefix}")
    return command


def run_visible(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: pathlib.Path = ROOT,
    capture: bool = False,
    filter_includes: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+", subprocess.list2cmdline(list(command)))
    if capture:
        result = run_capture(command, env=env, cwd=cwd)
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
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
            print(line, end="")
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
        print("configure: NVCC rejected this host compiler; retrying with -allow-unsupported-compiler")
        retry = run_visible(
            configure_command(args, environment, directory, allow_unsupported=True),
            env=environment.values,
            capture=True,
        )
        if retry.returncode == 0:
            return
    raise DevError("CMake configure failed.")


def build(args: argparse.Namespace, environment: Environment, directory: pathlib.Path) -> None:
    configure(args, environment, directory)
    assert environment.cmake
    result = run_visible(
        [str(environment.cmake.path), "--build", str(directory), "--parallel", str(args.jobs)],
        env=environment.values,
        filter_includes=environment.platform_name == "windows",
    )
    if result.returncode != 0:
        raise DevError("Build failed.")


def run_python_tests(environment: Environment) -> None:
    result = run_visible(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "dev_cli_test.py"],
        env=environment.values,
    )
    if result.returncode != 0:
        raise DevError("Developer CLI unit tests failed.")


def run_ctest(args: argparse.Namespace, environment: Environment, directory: pathlib.Path) -> None:
    assert environment.cmake
    ctest = environment.cmake.path.with_name("ctest.exe" if environment.platform_name == "windows" else "ctest")
    if not ctest.is_file():
        found = which("ctest.exe" if environment.platform_name == "windows" else "ctest", environment.values)
        if not found:
            raise DevError("CTest was not found.")
        ctest = found
    result = run_visible(
        [
            str(ctest),
            "--test-dir",
            str(directory),
            "--output-on-failure",
            "--parallel",
            str(args.jobs),
        ],
        env=environment.values,
    )
    if result.returncode != 0:
        raise DevError("CTest failed.")


def executable(directory: pathlib.Path, name: str, environment: Environment) -> pathlib.Path:
    suffix = ".exe" if environment.platform_name == "windows" else ""
    candidate = directory / f"{name}{suffix}"
    if candidate.is_file():
        return candidate
    raise DevError(f"Expected executable was not built: {name}{suffix}")


def run_smoke(args: argparse.Namespace, environment: Environment, directory: pathlib.Path) -> None:
    if environment.backend == "cuda":
        runner = executable(directory, "lfm25-run", environment)
        smoke_tests = [
            [str(runner), "--help"],
            [str(executable(directory, "c_api_smoke_test", environment))],
        ]
        if args.runtime_tests == "on":
            smoke_tests.append([str(executable(directory, "expert_residency_test", environment))])
    else:
        runner = executable(directory, "lfm25-cpu-run", environment)
        smoke_tests = [
            [str(runner), "--help"],
            [str(executable(directory, "cpu_c_api_smoke_test", environment))],
        ]

    for command in smoke_tests:
        result = run_visible(command, env=environment.values)
        if result.returncode != 0:
            raise DevError(f"Smoke command failed: {pathlib.Path(command[0]).name}")

    if environment.backend != "cuda":
        print("smoke: model inference skipped for CPU verification")
    elif not environment.checkpoint:
        print(f"smoke: {HF_REPO} is not cached; model inference skipped")
    elif not environment.gpu_name:
        print("smoke: no CUDA device is visible; model inference skipped")
    else:
        result = run_visible(
            [
                str(runner),
                "--repo",
                HF_REPO,
                "--prompt",
                "Hello",
                "--max-new-tokens",
                "4",
            ],
            env=environment.values,
        )
        if result.returncode != 0:
            raise DevError("Cached model inference smoke test failed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "build", "test", "smoke", "verify"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--build-type",
        choices=("Release", "RelWithDebInfo", "Debug"),
        default="RelWithDebInfo",
    )
    parser.add_argument(
        "--arch",
        type=normalize_arch,
        default="native",
        help="CUDA architecture, for example native, 86, or 120a",
    )
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--build-dir")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable doctor output")
    parser.add_argument(
        "--runtime-tests",
        choices=("on", "off"),
        default="on",
        help="Register and run tests that require a visible CUDA device",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.json and args.command != "doctor":
        parser.error("--json is only valid with doctor")
    return args


def normalize_arch(value: str) -> str:
    normalized = value.lower().removeprefix("compute_").removeprefix("sm_").replace(".", "")
    if normalized == "native":
        return normalized
    if not re.fullmatch(r"\d{2,3}[a-z]?", normalized):
        raise argparse.ArgumentTypeError(f"invalid CUDA architecture: {value}")
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    try:
        environment = discover_environment(args.backend, args.arch)
        print_doctor(environment, args.json)
        if args.command == "doctor":
            return 0 if environment.ok else 1
        if not environment.ok:
            return 1

        directory = build_directory(args, environment)
        build(args, environment, directory)
        if args.command in ("test", "verify"):
            run_python_tests(environment)
            run_ctest(args, environment, directory)
        if args.command in ("smoke", "verify"):
            run_smoke(args, environment, directory)
        print(f"{args.command}: PASS ({environment.backend}, {args.build_type}, {directory})")
        return 0
    except (DevError, OSError) as error:
        print(f"{args.command}: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
