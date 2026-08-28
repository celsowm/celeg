"""Environment, toolchain, CUDA, and checkpoint discovery."""

from __future__ import annotations

import dataclasses
import glob
import os
import pathlib
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

HF_REPO = "LiquidAI/LFM2.5-230M"
HF_GGUF_REPO = "LiquidAI/LFM2.5-350M-GGUF"
HF_GGUF_FILE = "LFM2.5-350M-Q4_K_M.gguf"
HF_MOE_REPO = "LiquidAI/LFM2.5-8B-A1B"
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
    macos_version: str = ""
    xcode_version: str = ""
    sdk_path: pathlib.Path | None = None
    metal_compiler: Tool | None = None
    metallib_compiler: Tool | None = None
    quantized_checkpoint: pathlib.Path | None = None
    moe_checkpoint: pathlib.Path | None = None

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
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def windows_oem_encoding() -> str:
    """Return the Python codec name for the console's active output codepage.

    cl.exe's localized /showIncludes note is written in whatever codepage the
    attached console session is actively using -- not necessarily UTF-8, and
    not necessarily the machine's static OEM codepage either. This can differ
    per terminal: a classic conhost/git-bash session typically inherits the
    OS default OEM codepage (e.g. cp850 on a pt-BR install), while a modern
    PowerShell 7 session commonly reconfigures its console to codepage 65001
    (UTF-8). GetConsoleOutputCP() reports whichever one is actually active;
    the static GetOEMCP() does not and silently double-encodes/mis-decodes
    output when the two diverge. Decoding cl.exe's output with the wrong
    codec corrupts every non-ASCII byte (either into U+FFFD, or -- if UTF-8
    bytes get mis-decoded as a single-byte codepage -- into unrelated
    mojibake), which then poisons Ninja's `msvc_deps_prefix` and silently
    disables MSVC header-dependency tracking for the whole build.
    """
    try:
        import ctypes

        codepage = ctypes.windll.kernel32.GetConsoleOutputCP()  # type: ignore[attr-defined]
        if not codepage:
            codepage = ctypes.windll.kernel32.GetOEMCP()  # type: ignore[attr-defined]
        if codepage:
            return f"cp{codepage}"
    except (AttributeError, OSError, ValueError):
        pass
    return "utf-8"


def run_capture_oem(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Like run_capture, but decodes output with the console OEM codepage.

    Use this instead of run_capture whenever a probed tool's output must be
    parsed for non-ASCII, locale-dependent text (see windows_oem_encoding).
    """
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=windows_oem_encoding(),
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


def xcrun_path(tool: str, env: Mapping[str, str], runner: Runner = run_capture) -> pathlib.Path | None:
    result = runner(["xcrun", "--find", tool], env=env)
    if result.returncode != 0:
        return None
    values = result.stdout.strip().splitlines()
    if not values:
        return None
    path = pathlib.Path(values[-1])
    return path.resolve() if path.is_file() else None


def command_output(command: Sequence[str], env: Mapping[str, str], runner: Runner) -> str:
    result = runner(command, env=env)
    return result.stdout.strip() if result.returncode == 0 else ""


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
    current = env_value(env, "PATH") or ""
    candidates = [str(path) for path in paths if path.is_dir()]
    candidates.extend(current.split(os.pathsep))
    seen: set[str] = set()
    entries: list[str] = []
    for entry in candidates:
        if not entry:
            continue
        normalized = os.path.normcase(entry.rstrip("\\/"))
        if normalized in seen:
            continue
        seen.add(normalized)
        entries.append(entry)
    if entries:
        key = next((name for name in env if name.casefold() == "path"), "PATH")
        env[key] = os.pathsep.join(entries)


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

    # CUDA 13.x supports the VS 2022 host compiler but rejects the preview
    # VS 2026 toolset that vswhere may otherwise report as "latest".
    program_files = env_value(env, "ProgramFiles")
    if program_files:
        vs_2022 = pathlib.Path(program_files) / "Microsoft Visual Studio" / "2022"
        candidates = sorted(vs_2022.glob("*/VC/Auxiliary/Build/vcvars64.bat"))
        if candidates:
            return candidates[-1].resolve()

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
            # VS 2022's environment scripts probe the active console.  Sending
            # their output to NUL makes that probe fail under redirected
            # processes, producing repeated Ctrl-C prompts and no toolchain.
            script.write(f'call "{vcvars}"\n')
            script.write("if errorlevel 1 exit /b %errorlevel%\n")
            script.write("set\n")
            script_path = pathlib.Path(script.name)
        result = runner(
            ["cmd.exe", "/d", "/c", str(script_path)],
            env=env,
        )
    finally:
        if script_path:
            script_path.unlink(missing_ok=True)
    if result.returncode != 0:
        detail = (result.stdout or "").strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
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


def fallback_msvc_environment(
    vcvars: pathlib.Path,
    env: Mapping[str, str],
) -> dict[str, str] | None:
    """Build the x64 MSVC environment when vcvars cannot run headlessly."""
    vc_root = vcvars.parents[2]
    vs_root = vc_root.parent
    tools_root = vc_root / "Tools" / "MSVC"
    toolsets = sorted(
        (path for path in tools_root.glob("*") if path.is_dir()),
        key=version_key,
        reverse=True,
    )
    if not toolsets:
        return None
    tools = toolsets[0]
    compiler_dir = tools / "bin" / "Hostx64" / "x64"
    if not (compiler_dir / "cl.exe").is_file():
        return None

    program_files_x86 = env_value(env, "ProgramFiles(x86)")
    if not program_files_x86:
        return None
    sdk_root = pathlib.Path(program_files_x86) / "Windows Kits" / "10"
    include_root = sdk_root / "Include"
    versions = sorted(
        (path for path in include_root.glob("*") if path.is_dir()),
        key=version_key,
        reverse=True,
    )
    if not versions:
        return None
    sdk_version = versions[0].name
    sdk_include = include_root / sdk_version
    sdk_lib = sdk_root / "Lib" / sdk_version
    required = [
        tools / "include",
        tools / "lib" / "x64",
        sdk_include / "ucrt",
        sdk_include / "um",
        sdk_include / "shared",
        sdk_lib / "ucrt" / "x64",
        sdk_lib / "um" / "x64",
    ]
    if not all(path.is_dir() for path in required):
        return None

    captured = dict(env)
    captured["VSINSTALLDIR"] = str(vs_root) + "\\"
    captured["VCINSTALLDIR"] = str(vc_root) + "\\"
    captured["VCToolsInstallDir"] = str(tools) + "\\"
    captured["VCToolsVersion"] = tools.name
    captured["WindowsSdkDir"] = str(sdk_root) + "\\"
    captured["WindowsSDKVersion"] = sdk_version + "\\"
    captured["INCLUDE"] = os.pathsep.join(
        str(path)
        for path in [
            tools / "include",
            tools / "ATLMFC" / "include",
            sdk_include / "ucrt",
            sdk_include / "um",
            sdk_include / "shared",
            sdk_include / "winrt",
            sdk_include / "cppwinrt",
        ]
        if path.is_dir()
    )
    captured["LIB"] = os.pathsep.join(
        str(path)
        for path in [
            tools / "ATLMFC" / "lib" / "x64",
            tools / "lib" / "x64",
            sdk_lib / "ucrt" / "x64",
            sdk_lib / "um" / "x64",
        ]
        if path.is_dir()
    )
    captured["LIBPATH"] = captured["LIB"]
    prepend_path(
        captured,
        [
            compiler_dir,
            sdk_root / "Bin" / sdk_version / "x64",
            sdk_root / "Bin" / "x64",
            vs_root / "Common7" / "IDE",
        ],
    )
    return captured


def detect_msvc_include_prefix(
    compiler: pathlib.Path,
    env: Mapping[str, str],
    runner: Runner = run_capture_oem,
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


def find_gguf_checkpoint(
    env: Mapping[str, str],
    repo: str = HF_GGUF_REPO,
    filename: str = HF_GGUF_FILE,
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
        model = candidate / filename
        if model.is_file():
            return model
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
    macos_version = ""
    xcode_version = ""
    sdk_path: pathlib.Path | None = None
    metal_compiler: Tool | None = None
    metallib_compiler: Tool | None = None

    msvc_include_prefix = ""
    if system == "windows":
        # VsDevCmd invokes vswhere.exe by name.  Some Windows installations
        # have NoDefaultCurrentDirectoryInExePath enabled, so relying on the
        # installer directory as the current directory is not sufficient.
        installer_root = env_value(values, "ProgramFiles(x86)")
        if installer_root:
            prepend_path(
                values,
                [
                    pathlib.Path(installer_root)
                    / "Microsoft Visual Studio"
                    / "Installer"
                ],
            )
        vcvars = discover_vcvars(values, runner)
        if vcvars:
            try:
                values = capture_vcvars(vcvars, values, runner)
                values["VSLANG"] = "1033"
            except DevError as error:
                fallback = fallback_msvc_environment(vcvars, values)
                if fallback is None:
                    errors.append(str(error))
                else:
                    values = fallback
                    warnings.append(
                        "vcvars64.bat could not initialize headlessly; using the discovered MSVC x64 toolchain directly."
                    )
        else:
            errors.append("Visual Studio C++ Build Tools were not found through the environment or vswhere.")

    cmake_path = which("cmake.exe" if system == "windows" else "cmake", values)
    ninja_path = which("ninja.exe" if system == "windows" else "ninja", values)
    compiler_name = "cl.exe" if system == "windows" else (env_value(values, "CXX") or "c++")
    compiler_path = which(compiler_name, values)

    cmake = Tool(cmake_path, executable_version(cmake_path, runner)) if cmake_path else None
    ninja = Tool(ninja_path, executable_version(ninja_path, runner)) if ninja_path else None
    compiler = Tool(compiler_path, executable_version(compiler_path, runner)) if compiler_path else None

    if system == "darwin":
        macos_version = command_output(["sw_vers", "-productVersion"], values, runner)
        sdk_value = command_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], values, runner)
        if sdk_value:
            sdk_path = pathlib.Path(sdk_value.splitlines()[-1]).resolve()
        xcode_text = command_output(["xcodebuild", "-version"], values, runner)
        xcode_match = re.search(r"^Xcode\s+(.+)$", xcode_text, re.MULTILINE)
        xcode_version = xcode_match.group(1).strip() if xcode_match else ""
        metal_path = xcrun_path("metal", values, runner)
        metallib_path = xcrun_path("metallib", values, runner)
        if metal_path:
            metal_compiler = Tool(metal_path, executable_version(metal_path, runner))
        if metallib_path:
            metallib_compiler = Tool(metallib_path, executable_version(metallib_path, runner))

    # CMake can use an absolute C/CXX compiler path while nvcc still invokes
    # the MSVC host compiler by the bare name `cl.exe` during CUDA compiler
    # identification.  Keep the selected toolchain self-contained in the
    # child environment so the official CUDA configure path behaves exactly
    # like the direct VS developer shell.
    if compiler and system == "windows":
        prepend_path(values, [compiler.path.parent])

    if not cmake:
        errors.append("CMake was not found on PATH.")
    if not ninja:
        errors.append("Ninja was not found on PATH.")
    if not compiler:
        errors.append("A C++ compiler was not found after environment initialization.")
    elif system == "windows":
        # Deliberately not the generic `runner`: probing cl.exe's localized
        # /showIncludes note needs OEM-codepage decoding (run_capture_oem),
        # not the UTF-8 decoding every other probe in this module expects.
        msvc_include_prefix = detect_msvc_include_prefix(compiler.path, values)
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

    metal_host = system == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    metal_buildable = bool(metal_host and cmake and ninja and compiler and sdk_path)
    if system == "darwin" and metal_host and not metal_compiler:
        warnings.append("Metal offline compiler was not found; runtime MSL compilation will be used.")

    backend = requested_backend
    if backend == "auto":
        backend = "cuda" if cuda_buildable else ("metal" if metal_buildable else "cpu")
        if candidates and not cuda_buildable:
            warnings.append("CUDA was found but no compatible concrete GPU architecture was detected; using CPU.")
    elif backend == "cuda":
        if not nvcc:
            errors.append("No CUDA toolkit supports the requested architecture.")
        if arch == "native":
            errors.append("CUDA architecture is native but no GPU compute capability was detected; pass --arch N.")
    elif backend == "metal":
        if system != "darwin":
            errors.append("Metal requires macOS.")
        elif not metal_host:
            errors.append("Metal requires an Apple Silicon host.")
        elif not sdk_path:
            errors.append("The macOS SDK was not found through xcrun.")

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
        checkpoint=find_checkpoint(values) or find_checkpoint(
            values, "LiquidAI/LFM2.5-350M"),
        errors=errors,
        warnings=warnings,
        macos_version=macos_version,
        xcode_version=xcode_version,
        sdk_path=sdk_path,
        metal_compiler=metal_compiler,
        metallib_compiler=metallib_compiler,
        quantized_checkpoint=find_gguf_checkpoint(values),
        moe_checkpoint=find_checkpoint(values, HF_MOE_REPO),
    )
