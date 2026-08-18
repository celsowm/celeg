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
    raise RuntimeError(f"missing fragment in {path}: {old[:100]!r}")


def replace_optional(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        file.write_text(text.replace(old, new), encoding="utf-8")


replace(
    "src/backend/cpu/detail/model_internal.hpp",
    "        std::vector<float> operator_norm;\n        std::vector<float> post_attention_norm;\n        std::vector<float> pre_feed_forward_norm;\n        std::vector<float> post_feed_forward_norm;\n        std::vector<float> per_layer_input_norm;\n        std::vector<float> ffn_norm;",
    "        std::vector<float> mixer_norm_before;\n        std::vector<float> mixer_norm_after;\n        std::vector<float> feed_forward_norm_before;\n        std::vector<float> feed_forward_norm_after;\n        std::vector<float> per_layer_input_norm;",
)

replace(
    "include/celeg/detail/model/layer_state.hpp",
    "    const __nv_bfloat16* operator_norm = nullptr;\n    const __nv_bfloat16* post_attention_norm = nullptr;\n    const __nv_bfloat16* ffn_norm = nullptr;\n    const __nv_bfloat16* post_feed_forward_norm = nullptr;",
    "    const __nv_bfloat16* mixer_norm_before = nullptr;\n    const __nv_bfloat16* mixer_norm_after = nullptr;\n    const __nv_bfloat16* feed_forward_norm_before = nullptr;\n    const __nv_bfloat16* feed_forward_norm_after = nullptr;",
)

for file in ROOT.rglob("*"):
    if not file.is_file() or file.suffix not in {".cpp", ".cu", ".cuh", ".hpp", ".h"}:
        continue
    text = file.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("common.operator_norm", "common.mixer_norm_before")
    text = text.replace("common.mixer_norm.after", "common.mixer_norm_after")
    text = text.replace("common.ffn_norm", "common.feed_forward_norm_before")
    text = text.replace("common.feed_forward_norm.after", "common.feed_forward_norm_after")
    text = text.replace("common_layer.operator_norm", "common_layer.mixer_norm_before")
    text = text.replace("common_layer.mixer_norm.after", "common_layer.mixer_norm_after")
    text = text.replace("common_layer.ffn_norm", "common_layer.feed_forward_norm_before")
    text = text.replace("common_layer.feed_forward_norm.after", "common_layer.feed_forward_norm_after")
    text = text.replace("common->operator_norm", "common->mixer_norm_before")
    text = text.replace("common->mixer_norm.after", "common->mixer_norm_after")
    text = text.replace("common->ffn_norm", "common->feed_forward_norm_before")
    text = text.replace("common->feed_forward_norm.after", "common->feed_forward_norm_after")
    file.write_text(text, encoding="utf-8")

replace(
    "src/backend/cpu/weights_loader.cpp",
    "    common.mixer_norm_before = load_norm(TensorRole::AttentionInputNorm,\n                                     layer_program.operator_norm);\n    if (std::holds_alternative<std::monostate>(layer_program.feed_forward)) {\n        common.feed_forward_norm_before = common.mixer_norm_before;\n        return common;\n    }\n    if (layer_program.mixer_norm.after) {\n        common.mixer_norm_after = load_norm(TensorRole::AttentionPostNorm,\n                                               *layer_program.mixer_norm.after);\n    }\n    common.feed_forward_norm_before = load_norm(TensorRole::FfnInputNorm,\n                                *layer_program.feed_forward_norm);\n    if (layer_program.feed_forward_norm.after) {\n        common.feed_forward_norm_after = load_norm(TensorRole::FfnOutputNorm,\n                                                  *layer_program.feed_forward_norm.after);\n    }",
    "    if (layer_program.mixer_norm.before) {\n        common.mixer_norm_before = load_norm(\n            TensorRole::AttentionInputNorm, *layer_program.mixer_norm.before);\n    }\n    if (layer_program.mixer_norm.after) {\n        common.mixer_norm_after = load_norm(\n            TensorRole::AttentionPostNorm, *layer_program.mixer_norm.after);\n    }\n    if (!std::holds_alternative<std::monostate>(layer_program.feed_forward)) {\n        if (layer_program.feed_forward_norm.before) {\n            common.feed_forward_norm_before = load_norm(\n                TensorRole::FfnInputNorm, *layer_program.feed_forward_norm.before);\n        }\n        if (layer_program.feed_forward_norm.after) {\n            common.feed_forward_norm_after = load_norm(\n                TensorRole::FfnOutputNorm, *layer_program.feed_forward_norm.after);\n        }\n    }",
)

replace(
    "src/backend/cpu/model_forward_token.cpp",
    "        cpu_rmsnorm(workspace_.hidden.data(), common.mixer_norm_before.data(), workspace_.normed.data(),\n                    shared->program.hidden, semantics.operator_norm.epsilon);",
    "        if (semantics.mixer_norm.before) {\n            cpu_rmsnorm(workspace_.hidden.data(), common.mixer_norm_before.data(),\n                        workspace_.normed.data(), shared->program.hidden,\n                        semantics.mixer_norm.before->epsilon);\n        } else {\n            std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.normed.begin());\n        }",
)
replace(
    "src/backend/cpu/model_forward_token.cpp",
    "            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.mixer_norm_after.data(),\n                                shared->program.hidden, semantics.mixer_norm.after->epsilon);",
    "            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.mixer_norm_after.data(),\n                                shared->program.hidden, semantics.mixer_norm.after->epsilon);",
)
replace(
    "src/backend/cpu/model_forward_token.cpp",
    "        cpu_rmsnorm(workspace_.hidden.data(), common.feed_forward_norm_before.data(), workspace_.normed.data(),\n                    shared->program.hidden, semantics.feed_forward_norm->epsilon);",
    "        if (semantics.feed_forward_norm.before) {\n            cpu_rmsnorm(workspace_.hidden.data(), common.feed_forward_norm_before.data(),\n                        workspace_.normed.data(), shared->program.hidden,\n                        semantics.feed_forward_norm.before->epsilon);\n        } else {\n            std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.normed.begin());\n        }",
)

replace_optional(
    "src/backend/cpu/model_forward_chunk.cpp",
    "        rmsnorm_rows(workspace_.chunk_hidden.data(), common.mixer_norm_before,\n                     workspace_.chunk_normed.data(), hidden,\n                     semantics.operator_norm.epsilon);",
    "        if (semantics.mixer_norm.before) {\n            rmsnorm_rows(workspace_.chunk_hidden.data(), common.mixer_norm_before,\n                         workspace_.chunk_normed.data(), hidden,\n                         semantics.mixer_norm.before->epsilon);\n        } else {\n            std::copy(workspace_.chunk_hidden.begin(), workspace_.chunk_hidden.end(),\n                      workspace_.chunk_normed.begin());\n        }",
)
replace_optional(
    "src/backend/cpu/model_forward_chunk.cpp",
    "        rmsnorm_rows(workspace_.chunk_hidden.data(), common.feed_forward_norm_before,\n                     workspace_.chunk_normed.data(), hidden,\n                     semantics.feed_forward_norm->epsilon);",
    "        if (semantics.feed_forward_norm.before) {\n            rmsnorm_rows(workspace_.chunk_hidden.data(), common.feed_forward_norm_before,\n                         workspace_.chunk_normed.data(), hidden,\n                         semantics.feed_forward_norm.before->epsilon);\n        } else {\n            std::copy(workspace_.chunk_hidden.begin(), workspace_.chunk_hidden.end(),\n                      workspace_.chunk_normed.begin());\n        }",
)

replace_optional(
    "src/backend/cpu/packed/execution.cpp",
    "            rmsnorm_rows(workspace_.hidden.data(), common.mixer_norm_before,\n                         workspace_.normed.data(), hidden);",
    "            if (layer_semantics.mixer_norm.before) {\n                rmsnorm_rows(workspace_.hidden.data(), common.mixer_norm_before,\n                             workspace_.normed.data(), hidden,\n                             layer_semantics.mixer_norm.before->epsilon);\n            } else {\n                std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.normed.begin());\n            }",
)
replace_optional(
    "src/backend/cpu/packed/execution.cpp",
    "            rmsnorm_rows(workspace_.hidden.data(), common.feed_forward_norm_before, workspace_.normed.data(), hidden);",
    "            if (layer_semantics.feed_forward_norm.before) {\n                rmsnorm_rows(workspace_.hidden.data(), common.feed_forward_norm_before,\n                             workspace_.normed.data(), hidden,\n                             layer_semantics.feed_forward_norm.before->epsilon);\n            } else {\n                std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.normed.begin());\n            }",
)

for path in [
    "src/backend/cpu/operators/recurrent.cpp",
    "src/backend/cpu/packed/execution.cpp",
]:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    text = text.replace(
        "shared.program.layers.at(layer).operator_norm.epsilon",
        "shared.program.layers.at(layer).mixer_norm.before.has_value()\n            ? shared.program.layers.at(layer).mixer_norm.before->epsilon\n            : shared.program.final_norm.epsilon")
    text = text.replace(
        "layer_semantics.operator_norm.epsilon",
        "layer_semantics.mixer_norm.before.has_value()\n                            ? layer_semantics.mixer_norm.before->epsilon\n                            : shared.program.final_norm.epsilon")
    file.write_text(text, encoding="utf-8")

replace(
    "src/backend/cuda/model/weight_setup.cpp",
    "        common_layer.mixer_norm_before = resources_.weight_loader_->load_rms_norm_weight(\n            repo, semantic_layer.operator_norm.weightless() ? std::string{} :\n                tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionInputNorm, i),\n            {resources_.program_.hidden}, semantic_layer.operator_norm.weight_kind);\n        if (!mixer_only_layer) {\n            common_layer.feed_forward_norm_before = load_norm(TensorRole::FfnInputNorm,\n                                              *semantic_layer.feed_forward_norm);\n        }\n        if (!mixer_only_layer && semantic_layer.mixer_norm.after.has_value()) {\n            common_layer.mixer_norm_after = load_norm(\n                TensorRole::AttentionPostNorm, *semantic_layer.mixer_norm.after);\n        }\n        if (!mixer_only_layer && semantic_layer.feed_forward_norm.after.has_value()) {\n            common_layer.feed_forward_norm_after = load_norm(\n                TensorRole::FfnOutputNorm, *semantic_layer.feed_forward_norm.after);\n        }",
    "        if (semantic_layer.mixer_norm.before) {\n            common_layer.mixer_norm_before = load_norm(\n                TensorRole::AttentionInputNorm, *semantic_layer.mixer_norm.before);\n        }\n        if (semantic_layer.mixer_norm.after) {\n            common_layer.mixer_norm_after = load_norm(\n                TensorRole::AttentionPostNorm, *semantic_layer.mixer_norm.after);\n        }\n        if (!mixer_only_layer && semantic_layer.feed_forward_norm.before) {\n            common_layer.feed_forward_norm_before = load_norm(\n                TensorRole::FfnInputNorm, *semantic_layer.feed_forward_norm.before);\n        }\n        if (!mixer_only_layer && semantic_layer.feed_forward_norm.after) {\n            common_layer.feed_forward_norm_after = load_norm(\n                TensorRole::FfnOutputNorm, *semantic_layer.feed_forward_norm.after);\n        }",
)

for file in ROOT.rglob("*"):
    if not file.is_file() or file.suffix not in {".cpp", ".cu", ".cuh", ".hpp", ".h"}:
        continue
    text = file.read_text(encoding="utf-8", errors="ignore")
    text = text.replace(".operator_norm.epsilon", ".mixer_norm.before->epsilon")
    text = text.replace(".feed_forward_norm->epsilon", ".feed_forward_norm.before->epsilon")
    file.write_text(text, encoding="utf-8")

print("backend semantic norm migration staged")
