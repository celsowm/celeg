from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def rewrite(path: str, fn) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    new = fn(text)
    if new == text:
        raise RuntimeError(f"no change produced for {path}")
    file.write_text(new, encoding="utf-8")


def simple(path: str, pairs: list[tuple[str, str]], required: bool = True) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    changed = False
    for old, new in pairs:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if required and not changed:
        raise RuntimeError(f"no expected fragment found in {path}")
    file.write_text(text, encoding="utf-8")


simple("src/backend/cpu/model_forward_token.cpp", [
    ('''        if (semantics.residual.multiplier != 1.0f) {\n            for (float& value : workspace_.hidden) value *= semantics.residual.multiplier;\n        }\n        if (semantics.mixer_norm.after.has_value()) {\n            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.mixer_norm_after.data(),\n                                shared->program.hidden, semantics.mixer_norm.after->epsilon);\n        }''',
     '''        if (semantics.mixer_norm.after.has_value()) {\n            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.mixer_norm_after.data(),\n                                shared->program.hidden, semantics.mixer_norm.after->epsilon);\n        }\n        if (semantics.residual.multiplier != 1.0f) {\n            for (float& value : workspace_.hidden) value *= semantics.residual.multiplier;\n        }'''),
    ('''            if (semantics.residual.multiplier != 1.0f) {\n                for (float& value : workspace_.mlp_output) value *= semantics.residual.multiplier;\n            }\n            if (semantics.feed_forward_norm.after.has_value()) {\n                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.feed_forward_norm_after.data(),\n                                    shared->program.hidden, semantics.feed_forward_norm.after->epsilon);\n            }''',
     '''            if (semantics.feed_forward_norm.after.has_value()) {\n                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.feed_forward_norm_after.data(),\n                                    shared->program.hidden, semantics.feed_forward_norm.after->epsilon);\n            }\n            if (semantics.residual.multiplier != 1.0f) {\n                for (float& value : workspace_.mlp_output) value *= semantics.residual.multiplier;\n            }'''),
])

simple("src/backend/cpu/model_forward_chunk.cpp", [
    ('''        if (semantics.residual.multiplier != 1.0f) {\n            scale(workspace_.chunk_hidden, rows * hidden,\n                  semantics.residual.multiplier);\n        }\n        if (semantics.mixer_norm.after.has_value()) {\n            rmsnorm_rows_inplace(workspace_.chunk_hidden.data(), common.mixer_norm_after, hidden,\n                                 semantics.mixer_norm.after->epsilon);\n        }''',
     '''        if (semantics.mixer_norm.after.has_value()) {\n            rmsnorm_rows_inplace(workspace_.chunk_hidden.data(), common.mixer_norm_after, hidden,\n                                 semantics.mixer_norm.after->epsilon);\n        }\n        if (semantics.residual.multiplier != 1.0f) {\n            scale(workspace_.chunk_hidden, rows * hidden,\n                  semantics.residual.multiplier);\n        }'''),
    ('''        if (semantics.residual.multiplier != 1.0f) {\n            scale(workspace_.chunk_mlp, rows * hidden,\n                  semantics.residual.multiplier);\n        }\n        if (semantics.feed_forward_norm.after.has_value()) {\n            rmsnorm_rows_inplace(workspace_.chunk_mlp.data(), common.feed_forward_norm_after, hidden,\n                                 semantics.feed_forward_norm.after->epsilon);\n        }''',
     '''        if (semantics.feed_forward_norm.after.has_value()) {\n            rmsnorm_rows_inplace(workspace_.chunk_mlp.data(), common.feed_forward_norm_after, hidden,\n                                 semantics.feed_forward_norm.after->epsilon);\n        }\n        if (semantics.residual.multiplier != 1.0f) {\n            scale(workspace_.chunk_mlp, rows * hidden,\n                  semantics.residual.multiplier);\n        }'''),
])

simple("src/backend/cpu/packed/execution.cpp", [
    ('''            if (layer_semantics.residual.multiplier != 1.0f) {\n                rows_for([&](size_t row) {\n                    float* values = workspace_.hidden.data() + row * hidden;\n                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;\n                });\n            }\n            if (layer_semantics.mixer_norm.after.has_value()) {\n                rmsnorm_rows_inplace(workspace_.hidden.data(), common.mixer_norm_after, hidden);\n            }''',
     '''            if (layer_semantics.mixer_norm.after.has_value()) {\n                rmsnorm_rows_inplace(workspace_.hidden.data(), common.mixer_norm_after, hidden,\n                                     layer_semantics.mixer_norm.after->epsilon);\n            }\n            if (layer_semantics.residual.multiplier != 1.0f) {\n                rows_for([&](size_t row) {\n                    float* values = workspace_.hidden.data() + row * hidden;\n                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;\n                });\n            }'''),
    ('''            if (layer_semantics.residual.multiplier != 1.0f) {\n                rows_for([&](size_t row) {\n                    float* values = workspace_.mlp_output.data() + row * hidden;\n                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;\n                });\n            }\n            if (layer_semantics.feed_forward_norm.after.has_value()) {\n                rmsnorm_rows_inplace(workspace_.mlp_output.data(), common.feed_forward_norm_after, hidden);\n            }''',
     '''            if (layer_semantics.feed_forward_norm.after.has_value()) {\n                rmsnorm_rows_inplace(workspace_.mlp_output.data(), common.feed_forward_norm_after, hidden,\n                                     layer_semantics.feed_forward_norm.after->epsilon);\n            }\n            if (layer_semantics.residual.multiplier != 1.0f) {\n                rows_for([&](size_t row) {\n                    float* values = workspace_.mlp_output.data() + row * hidden;\n                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;\n                });\n            }'''),
])

# Token CUDA: remove mixer-local residual scaling, then add it once at the layer boundary.
path = ROOT / "src/backend/cuda/model/decode.cpp"
text = path.read_text(encoding="utf-8")
text = re.sub(r'\n\s*launch_scale\(workspace_\.hidden_\.data\(\), resources_\.program_\.hidden,\n\s*semantics\.residual\.multiplier(?:,\n\s*stream_\.get\(\)|, stream_\.get\(\));\n', '\n', text)
old = '''    if (semantics.mixer_norm.after) {\n        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,\n                       workspace_.hidden_.data(), 1, resources_.program_.hidden,\n                       semantics.mixer_norm.after->epsilon, stream_.get());\n    }\n    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),'''
new = '''    if (semantics.mixer_norm.after) {\n        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,\n                       workspace_.hidden_.data(), 1, resources_.program_.hidden,\n                       semantics.mixer_norm.after->epsilon, stream_.get());\n    }\n    if (semantics.residual.multiplier != 1.0f) {\n        launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,\n                     semantics.residual.multiplier, stream_.get());\n    }\n    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),'''
if old not in text:
    raise RuntimeError("CUDA token layer boundary not found")
text = text.replace(old, new, 1)
text = text.replace(
    '''    const float epsilon = kv.paged() ? semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon : resources_.program_.final_norm.epsilon\n                                     : semantics.mixer_norm.after->epsilon;''',
    '''    const float epsilon = semantics.mixer_norm.before\n        ? semantics.mixer_norm.before->epsilon\n        : resources_.program_.final_norm.epsilon;''')
path.write_text(text, encoding="utf-8")

# Graph-captured CUDA decode follows the same boundary rule.
simple("src/backend/cuda/model/execution.cu", [
    ('''        if (common_layer.mixer_norm_after) {\n            launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,\n                           workspace_.hidden_.data(), 1, resources_.program_.hidden,\n                           semantics.mixer_norm.after->epsilon, stream_.get());\n        }\n        decode_phase_profile().begin(stream_.get());''',
     '''        if (common_layer.mixer_norm_after) {\n            launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,\n                           workspace_.hidden_.data(), 1, resources_.program_.hidden,\n                           semantics.mixer_norm.after->epsilon, stream_.get());\n        }\n        if (semantics.residual.multiplier != 1.0f) {\n            launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,\n                         semantics.residual.multiplier, stream_.get());\n        }\n        decode_phase_profile().begin(stream_.get());'''),
])

# Enqueued attention must write pure mixer output: no beta residual fusion and no local scaling.
path = ROOT / "src/backend/cuda/model/attention_execution.cu"
text = path.read_text(encoding="utf-8")
text = re.sub(r'resources_\.options_\.fused_residuals\s*&&\s*!common_layer\.mixer_norm_after\s*&&\s*std::holds_alternative<std::monostate>\(semantics\.feed_forward\)\s*\?\s*0\.0f\s*:\s*1\.0f', '0.0f', text)
text = re.sub(r'\n\s*launch_scale\(workspace_\.hidden_\.data\(\), resources_\.program_\.hidden,\n\s*semantics\.residual\.multiplier(?:,\n\s*stream_\.get\(\)|, stream_\.get\(\));', '', text)
path.write_text(text, encoding="utf-8")

# Batched prefill: pure operators, one layer-owned postnorm/scale/residual boundary.
simple("src/backend/cuda/model/prefill_batched.cpp", [
    ('''    if (common_layer.mixer_norm_after) {\n        launch_rmsnorm(\n            workspace.prefill_hidden_.data(), common_layer.mixer_norm_after,\n            workspace.prefill_hidden_.data(),\n            rows, hidden, semantics.mixer_norm.after->epsilon,\n            model.stream_.get());\n    }\n\n    prof.begin(model.stream_.get());''',
     '''    if (common_layer.mixer_norm_after) {\n        launch_rmsnorm(\n            workspace.prefill_hidden_.data(), common_layer.mixer_norm_after,\n            workspace.prefill_hidden_.data(),\n            rows, hidden, semantics.mixer_norm.after->epsilon,\n            model.stream_.get());\n    }\n    if (semantics.residual.multiplier != 1.0f) {\n        launch_scale(workspace.prefill_hidden_.data(), rows * hidden,\n                     semantics.residual.multiplier, model.stream_.get());\n    }\n\n    prof.begin(model.stream_.get());'''),
])

path = ROOT / "src/backend/cuda/model/prefill_batched/attention/regular.cpp"
text = path.read_text(encoding="utf-8")
text = text.replace(
    '''    const bool output_gate = layout.output_gate.has_value();\n    const bool gate_packed = output_gate && layout.output_gate->packed_with_query;''',
    '''    const bool output_gate = layout.output_gate.has_value();\n    const bool gate_packed = output_gate && layout.output_gate->packed_with_query;\n    const float qk_norm_epsilon = layout.query_norm\n        ? layout.query_norm->epsilon\n        : (semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon\n                                       : model.resources_.program_.final_norm.epsilon);''')
text = text.replace("layout.query_norm->epsilon,", "qk_norm_epsilon,")
text = re.sub(
    r'rows, hidden, layout\.query_width\(\),\s*model\.resources_\.options_\.fused_residuals\s*&&\s*!common_layer\.mixer_norm_after\s*\?\s*1\.0f\s*:\s*0\.0f\);\s*launch_scale\(\s*workspace\.prefill_hidden_\.data\(\), rows \* hidden,\s*semantics\.residual\.multiplier, model\.stream_\.get\(\)\);',
    'rows, hidden, layout.query_width(), 0.0f);', text)
path.write_text(text, encoding="utf-8")

path = ROOT / "src/backend/cuda/model/prefill_batched/non_attention.cpp"
text = path.read_text(encoding="utf-8")
text = re.sub(r'\n\s*launch_scale\(model\.workspace_\.prefill_hidden_\.data\(\), rows \* model\.resources_\.program_\.hidden,\n\s*semantics\.residual\.multiplier, model\.stream_\.get\(\)\);', '', text)
path.write_text(text, encoding="utf-8")

# Packed CUDA follows the same rule.
simple("src/backend/cuda/packed/layer_executor.cu", [
    ('''        if (semantics.mixer_norm.after) {\n            launch_rmsnorm(workspace_.hidden.data(), common_layer.mixer_norm_after,\n                           workspace_.hidden.data(), rows, workspace_.program_.hidden,\n                           semantics.mixer_norm.after->epsilon, workspace_.stream.get());\n        }\n        launch_residual_add(workspace_.hidden.data(), workspace_.residual.data(),''',
     '''        if (semantics.mixer_norm.after) {\n            launch_rmsnorm(workspace_.hidden.data(), common_layer.mixer_norm_after,\n                           workspace_.hidden.data(), rows, workspace_.program_.hidden,\n                           semantics.mixer_norm.after->epsilon, workspace_.stream.get());\n        }\n        if (semantics.residual.multiplier != 1.0f) {\n            launch_scale(workspace_.hidden.data(), rows * workspace_.program_.hidden,\n                         semantics.residual.multiplier, workspace_.stream.get());\n        }\n        launch_residual_add(workspace_.hidden.data(), workspace_.residual.data(),'''),
])

path = ROOT / "src/backend/cuda/packed/operators.cu"
text = path.read_text(encoding="utf-8")
text = text.replace("layout.query_norm->epsilon, layout.has_query_key_norm(),",
                    "layout.query_norm ? layout.query_norm->epsilon : reference.program().final_norm.epsilon, layout.has_query_key_norm(),")
# First residual scale in this file belongs to PackedAttentionExecutor and moves to layer_executor.
attention_scale = '''    launch_scale(w.hidden.data(), rows * context.program.hidden,\n                 semantics.residual.multiplier, w.stream.get());\n'''
if attention_scale in text:
    text = text.replace(attention_scale, "", 1)
for buffer in ["w.mlp_output.data()", "w.moe_output.data()"]:
    scale = f'''    launch_scale({buffer}, rows * context.program.hidden,\n                 semantics.residual.multiplier, w.stream.get());\n'''
    norm_prefix = f'''    if (semantics.feed_forward_norm.after) {{\n        launch_rmsnorm({buffer}, common_layer.feed_forward_norm_after,\n                       {buffer}, rows, context.program.hidden,\n                       semantics.feed_forward_norm.after->epsilon, w.stream.get());\n    }}\n'''
    if scale in text and norm_prefix in text:
        text = text.replace(scale + norm_prefix, norm_prefix + '''    if (semantics.residual.multiplier != 1.0f) {\n''' + scale.replace("    launch_scale", "        launch_scale") + '''    }\n''', 1)
path.write_text(text, encoding="utf-8")

# MoE residency outputs are normalized before the residual multiplier.
path = ROOT / "src/backend/cuda/model/residency.cu"
text = path.read_text(encoding="utf-8")
for buffer, hidden_base in [
    ("workspace_.moe_output_.data()", "workspace_.hidden_.data()"),
    ("workspace_.moe_pf_output_.data()", "workspace_.prefill_hidden_.data()"),
]:
    scale_pattern = re.compile(rf'\s*if \(layer_semantics\.residual\.multiplier != 1\.0f\) \{{\s*launch_scale\({re.escape(buffer)}, ([^;]+);\s*\}}')
    match = scale_pattern.search(text)
    if not match:
        continue
    scale_block = match.group(0).strip() + "\n"
    text = text[:match.start()] + "\n" + text[match.end():]
    marker = f"    launch_residual_add({hidden_base}, {buffer},"
    pos = text.find(marker)
    if pos >= 0:
        text = text[:pos] + "    " + scale_block.replace("\n", "\n    ").rstrip() + "\n" + text[pos:]
path.write_text(text, encoding="utf-8")

# Sanity checks for the core canonical order.
checks = {
    "src/backend/cpu/model_forward_token.cpp": "semantics.mixer_norm.after.has_value()",
    "src/backend/cuda/model/decode.cpp": "if (semantics.residual.multiplier != 1.0f)",
    "src/backend/cuda/model/prefill_batched.cpp": "semantics.residual.multiplier, model.stream_.get()",
    "src/backend/cuda/packed/layer_executor.cu": "semantics.residual.multiplier, workspace_.stream.get()",
}
for rel, needle in checks.items():
    if needle not in (ROOT / rel).read_text(encoding="utf-8"):
        raise RuntimeError(f"canonical residual boundary missing from {rel}")

print("exact post-norm residual cleanup staged")
