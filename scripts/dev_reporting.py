"""Doctor report serialization and terminal presentation."""

from __future__ import annotations

import json

from dev_environment import Environment, Tool


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

