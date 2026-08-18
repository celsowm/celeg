from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        file.write_text(text.replace(old, new), encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"missing fragment in {path}: {old[:120]!r}")


replace(
    "src/backend/cpu/operators/recurrent.cpp",
    "    shared.linear.gemv(weights.out, workspace.op_output.data(),\n                       workspace.hidden.data());\n    cpu_residual_add(workspace.hidden.data(), workspace.residual.data(),\n                     shared.program.hidden);",
    "    shared.linear.gemv(weights.out, workspace.op_output.data(),\n                       workspace.hidden.data());",
)

replace(
    "src/backend/cpu/operators/feed_forward.cpp",
    "    shared.linear.gemv(weights.w2, workspace.activated.data(),\n                       workspace.hidden.data());\n    cpu_residual_add(workspace.hidden.data(), workspace.residual.data(),\n                     shared.program.hidden);",
    "    shared.linear.gemv(weights.w2, workspace.activated.data(),\n                       workspace.hidden.data());",
)

replace(
    "src/backend/cpu/model_forward_token.cpp",
    "        bool mixer_owns_layer = false;\n",
    "",
)
replace(
    "src/backend/cpu/model_forward_token.cpp",
    "          [&](const CpuCompiledModel::Mamba2Weights* mamba) {\n            execute_cpu_mamba2_token(execution, recurrent_state, index, *mamba);\n            mixer_owns_layer = true;\n          },",
    "          [&](const CpuCompiledModel::Mamba2Weights* mamba) {\n            execute_cpu_mamba2_token(execution, recurrent_state, index, *mamba);\n          },",
)
replace(
    "src/backend/cpu/model_forward_token.cpp",
    "            execute_cpu_mlp_only_token(execution, index, *mlp);\n            mixer_owns_layer = true;\n          });",
    "            execute_cpu_mlp_only_token(execution, index, *mlp);\n          });",
)
replace(
    "src/backend/cpu/model_forward_token.cpp",
    "        if (compute_logits && mixer_owns_layer) {\n            celeg_debug_dump_hidden((\"layer_\" + std::to_string(index)).c_str(),\n                                    workspace_.hidden.data(), shared->program.hidden);\n        }\n        if (mixer_owns_layer) continue;\n",
    "",
)

print("CPU mixer residual ownership unified")
