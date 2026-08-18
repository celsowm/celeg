from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def edit(path: str, old: str, new: str, *, all_occurrences: bool = False) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        text = text.replace(old, new) if all_occurrences else text.replace(old, new, 1)
        file.write_text(text, encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"missing fragment in {path}: {old[:120]!r}")


def optional(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        file.write_text(text.replace(old, new), encoding="utf-8")


# CPU: sublayer output -> post norm -> residual multiplier -> residual add.
for path, value in [
    ("src/backend/cpu/model_forward_token.cpp", "workspace_.hidden"),
    ("src/backend/cpu/model_forward_chunk.cpp", "workspace_.chunk_hidden"),
]:
    pass

edit(
    "src/backend/cpu/model_forward_token.cpp",
    '''        if (semantics.residual.multiplier != 1.0f) {
            cpu_scale(workspace_.hidden.data(), shared->program.hidden, semantics.residual.multiplier);
        }
        if (semantics.mixer_norm.after) {
            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.mixer_norm_after.data(),
                                shared->program.hidden, semantics.mixer_norm.after->epsilon);
        }''',
    '''        if (semantics.mixer_norm.after) {
            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.mixer_norm_after.data(),
                                shared->program.hidden, semantics.mixer_norm.after->epsilon);
        }
        if (semantics.residual.multiplier != 1.0f) {
            cpu_scale(workspace_.hidden.data(), shared->program.hidden, semantics.residual.multiplier);
        }''')
edit(
    "src/backend/cpu/model_forward_token.cpp",
    '''            if (semantics.residual.multiplier != 1.0f) {
                cpu_scale(workspace_.mlp_output.data(), shared->program.hidden, semantics.residual.multiplier);
            }
            if (semantics.feed_forward_norm.after) {
                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.feed_forward_norm_after.data(),
                                    shared->program.hidden, semantics.feed_forward_norm.after->epsilon);
            }''',
    '''            if (semantics.feed_forward_norm.after) {
                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.feed_forward_norm_after.data(),
                                    shared->program.hidden, semantics.feed_forward_norm.after->epsilon);
            }
            if (semantics.residual.multiplier != 1.0f) {
                cpu_scale(workspace_.mlp_output.data(), shared->program.hidden, semantics.residual.multiplier);
            }''',
    all_occurrences=True)

edit(
    "src/backend/cpu/model_forward_chunk.cpp",
    '''        if (semantics.residual.multiplier != 1.0f) {
            scale(workspace_.chunk_hidden, rows * hidden,
                  semantics.residual.multiplier);
        }
        if (semantics.mixer_norm.after.has_value()) {
            rmsnorm_rows_inplace(workspace_.chunk_hidden.data(), common.mixer_norm_after, hidden,
                                 semantics.mixer_norm.after->epsilon);
        }''',
    '''        if (semantics.mixer_norm.after.has_value()) {
            rmsnorm_rows_inplace(workspace_.chunk_hidden.data(), common.mixer_norm_after, hidden,
                                 semantics.mixer_norm.after->epsilon);
        }
        if (semantics.residual.multiplier != 1.0f) {
            scale(workspace_.chunk_hidden, rows * hidden,
                  semantics.residual.multiplier);
        }''')
edit(
    "src/backend/cpu/model_forward_chunk.cpp",
    '''        if (semantics.residual.multiplier != 1.0f) {
            scale(workspace_.chunk_mlp, rows * hidden,
                  semantics.residual.multiplier);
        }
        if (semantics.feed_forward_norm.after.has_value()) {
            rmsnorm_rows_inplace(workspace_.chunk_mlp.data(), common.feed_forward_norm_after, hidden,
                                 semantics.feed_forward_norm.after->epsilon);
        }''',
    '''        if (semantics.feed_forward_norm.after.has_value()) {
            rmsnorm_rows_inplace(workspace_.chunk_mlp.data(), common.feed_forward_norm_after, hidden,
                                 semantics.feed_forward_norm.after->epsilon);
        }
        if (semantics.residual.multiplier != 1.0f) {
            scale(workspace_.chunk_mlp, rows * hidden,
                  semantics.residual.multiplier);
        }''')

edit(
    "src/backend/cpu/packed/execution.cpp",
    '''            if (layer_semantics.residual.multiplier != 1.0f) {
                rows_for([&](size_t row) {
                    float* values = workspace_.hidden.data() + row * hidden;
                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;
                });
            }
            if (layer_semantics.mixer_norm.after.has_value()) {
                rmsnorm_rows_inplace(workspace_.hidden.data(), common.mixer_norm_after, hidden);
            }''',
    '''            if (layer_semantics.mixer_norm.after.has_value()) {
                rmsnorm_rows_inplace(workspace_.hidden.data(), common.mixer_norm_after, hidden,
                                     layer_semantics.mixer_norm.after->epsilon);
            }
            if (layer_semantics.residual.multiplier != 1.0f) {
                rows_for([&](size_t row) {
                    float* values = workspace_.hidden.data() + row * hidden;
                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;
                });
            }''')
edit(
    "src/backend/cpu/packed/execution.cpp",
    '''            if (layer_semantics.residual.multiplier != 1.0f) {
                rows_for([&](size_t row) {
                    float* values = workspace_.mlp_output.data() + row * hidden;
                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;
                });
            }
            if (layer_semantics.feed_forward_norm.after.has_value()) {
                rmsnorm_rows_inplace(workspace_.mlp_output.data(), common.feed_forward_norm_after, hidden);
            }''',
    '''            if (layer_semantics.feed_forward_norm.after.has_value()) {
                rmsnorm_rows_inplace(workspace_.mlp_output.data(), common.feed_forward_norm_after, hidden,
                                     layer_semantics.feed_forward_norm.after->epsilon);
            }
            if (layer_semantics.residual.multiplier != 1.0f) {
                rows_for([&](size_t row) {
                    float* values = workspace_.mlp_output.data() + row * hidden;
                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;
                });
            }''')

# CUDA token paths: mixers always produce a pure sublayer output. The layer owns scale/residual.
edit(
    "src/backend/cuda/model/decode.cpp",
    '''    if (semantics.mixer_norm.after) {
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,
                       workspace_.hidden_.data(), 1, resources_.program_.hidden,
                       semantics.mixer_norm.after->epsilon, stream_.get());
    }
    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),''',
    '''    if (semantics.mixer_norm.after) {
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,
                       workspace_.hidden_.data(), 1, resources_.program_.hidden,
                       semantics.mixer_norm.after->epsilon, stream_.get());
    }
    if (semantics.residual.multiplier != 1.0f) {
        launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                     semantics.residual.multiplier, stream_.get());
    }
    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),''')
optional(
    "src/backend/cuda/model/decode.cpp",
    '''    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier,
                 stream_.get());
''',
    "")
optional(
    "src/backend/cuda/model/decode.cpp",
    '''    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
''',
    "")
edit(
    "src/backend/cuda/model/decode.cpp",
    '''    const float epsilon = kv.paged() ? semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon : resources_.program_.final_norm.epsilon
                                     : semantics.mixer_norm.after->epsilon;''',
    '''    const float epsilon = semantics.mixer_norm.before
        ? semantics.mixer_norm.before->epsilon
        : resources_.program_.final_norm.epsilon;''')

edit(
    "src/backend/cuda/model/execution.cu",
    '''        if (common_layer.mixer_norm_after) {
            launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,
                           workspace_.hidden_.data(), 1, resources_.program_.hidden,
                           semantics.mixer_norm.after->epsilon, stream_.get());
        }
        decode_phase_profile().begin(stream_.get());''',
    '''        if (common_layer.mixer_norm_after) {
            launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,
                           workspace_.hidden_.data(), 1, resources_.program_.hidden,
                           semantics.mixer_norm.after->epsilon, stream_.get());
        }
        if (semantics.residual.multiplier != 1.0f) {
            launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                         semantics.residual.multiplier, stream_.get());
        }
        decode_phase_profile().begin(stream_.get());''')

# Legacy/enqueued CUDA attention path must not fuse or scale the residual branch itself.
file = ROOT / "src/backend/cuda/model/attention_execution.cu"
text = file.read_text(encoding="utf-8")
text = text.replace(
    '''resources_.options_.fused_residuals && !common_layer.mixer_norm_after &&
                           std::holds_alternative<std::monostate>(semantics.feed_forward) ? 0.0f : 1.0f''',
    "0.0f")
text = text.replace(
    '''resources_.options_.fused_residuals && !common_layer.mixer_norm_after &&
                       std::holds_alternative<std::monostate>(semantics.feed_forward) ? 0.0f : 1.0f''',
    "0.0f")
text = text.replace(
    '''                    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                                 semantics.residual.multiplier, stream_.get());
''', "")
text = text.replace(
    '''                launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                             semantics.residual.multiplier,
                             stream_.get());
''', "")
text = text.replace(
    '''            launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                         semantics.residual.multiplier,
                         stream_.get());
''', "")
file.write_text(text, encoding="utf-8")

# Batched prefill uses the same canonical layer-owned residual sequence.
edit(
    "src/backend/cuda/model/prefill_batched.cpp",
    '''    if (common_layer.mixer_norm_after) {
        launch_rmsnorm(
            workspace.prefill_hidden_.data(), common_layer.mixer_norm_after,
            workspace.prefill_hidden_.data(),
            rows, hidden, semantics.mixer_norm.after->epsilon,
            model.stream_.get());
    }

    prof.begin(model.stream_.get());''',
    '''    if (common_layer.mixer_norm_after) {
        launch_rmsnorm(
            workspace.prefill_hidden_.data(), common_layer.mixer_norm_after,
            workspace.prefill_hidden_.data(),
            rows, hidden, semantics.mixer_norm.after->epsilon,
            model.stream_.get());
    }
    if (semantics.residual.multiplier != 1.0f) {
        launch_scale(workspace.prefill_hidden_.data(), rows * hidden,
                     semantics.residual.multiplier, model.stream_.get());
    }

    prof.begin(model.stream_.get());''')

edit(
    "src/backend/cuda/model/prefill_batched/attention/regular.cpp",
    '''    const bool output_gate = layout.output_gate.has_value();
    const bool gate_packed = output_gate && layout.output_gate->packed_with_query;''',
    '''    const bool output_gate = layout.output_gate.has_value();
    const bool gate_packed = output_gate && layout.output_gate->packed_with_query;
    const float qk_norm_epsilon = layout.query_norm
        ? layout.query_norm->epsilon
        : (semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon
                                       : model.resources_.program_.final_norm.epsilon);''')
optional(
    "src/backend/cuda/model/prefill_batched/attention/regular.cpp",
    "layout.query_norm->epsilon,",
    "qk_norm_epsilon,")
edit(
    "src/backend/cuda/model/prefill_batched/attention/regular.cpp",
    '''        rows, hidden, layout.query_width(),
        model.resources_.options_.fused_residuals && !common_layer.mixer_norm_after
            ? 1.0f : 0.0f);
    launch_scale(
        workspace.prefill_hidden_.data(), rows * hidden,
        semantics.residual.multiplier, model.stream_.get());''',
    '''        rows, hidden, layout.query_width(), 0.0f);''')
optional(
    "src/backend/cuda/model/prefill_batched/non_attention.cpp",
    '''    launch_scale(model.workspace_.prefill_hidden_.data(), rows * model.resources_.program_.hidden,
                 semantics.residual.multiplier, model.stream_.get());
''',
    "")

# Packed CUDA: operator output is pure; layer owns post norm + multiplier + residual.
edit(
    "src/backend/cuda/packed/layer_executor.cu",
    '''        if (semantics.mixer_norm.after) {
            launch_rmsnorm(workspace_.hidden.data(), common_layer.mixer_norm_after,
                           workspace_.hidden.data(), rows, workspace_.program_.hidden,
                           semantics.mixer_norm.after->epsilon, workspace_.stream.get());
        }
        launch_residual_add(workspace_.hidden.data(), workspace_.residual.data(),''',
    '''        if (semantics.mixer_norm.after) {
            launch_rmsnorm(workspace_.hidden.data(), common_layer.mixer_norm_after,
                           workspace_.hidden.data(), rows, workspace_.program_.hidden,
                           semantics.mixer_norm.after->epsilon, workspace_.stream.get());
        }
        if (semantics.residual.multiplier != 1.0f) {
            launch_scale(workspace_.hidden.data(), rows * workspace_.program_.hidden,
                         semantics.residual.multiplier, workspace_.stream.get());
        }
        launch_residual_add(workspace_.hidden.data(), workspace_.residual.data(),''')

file = ROOT / "src/backend/cuda/packed/operators.cu"
text = file.read_text(encoding="utf-8")
text = text.replace(
    '''            static_cast<float>(rope->rotary_fraction),
            layout.query_norm->epsilon, layout.has_query_key_norm(),''',
    '''            static_cast<float>(rope->rotary_fraction),
            layout.query_norm ? layout.query_norm->epsilon : reference.program().final_norm.epsilon,
            layout.has_query_key_norm(),''')
text = text.replace(
    '''    launch_scale(w.hidden.data(), rows * context.program.hidden,
                 semantics.residual.multiplier, w.stream.get());
''', "", 1)
text = text.replace(
    '''    launch_scale(w.mlp_output.data(), rows * context.program.hidden,
                 semantics.residual.multiplier, w.stream.get());
    if (semantics.feed_forward_norm.after) {
        launch_rmsnorm(w.mlp_output.data(), common_layer.feed_forward_norm_after,
                       w.mlp_output.data(), rows, context.program.hidden,
                       semantics.feed_forward_norm.after->epsilon, w.stream.get());
    }''',
    '''    if (semantics.feed_forward_norm.after) {
        launch_rmsnorm(w.mlp_output.data(), common_layer.feed_forward_norm_after,
                       w.mlp_output.data(), rows, context.program.hidden,
                       semantics.feed_forward_norm.after->epsilon, w.stream.get());
    }
    if (semantics.residual.multiplier != 1.0f) {
        launch_scale(w.mlp_output.data(), rows * context.program.hidden,
                     semantics.residual.multiplier, w.stream.get());
    }''')
text = text.replace(
    '''    launch_scale(w.moe_output.data(), rows * context.program.hidden,
                 semantics.residual.multiplier, w.stream.get());
    if (semantics.feed_forward_norm.after) {
        launch_rmsnorm(w.moe_output.data(), common_layer.feed_forward_norm_after,
                       w.moe_output.data(), rows, context.program.hidden,
                       semantics.feed_forward_norm.after->epsilon, w.stream.get());
    }''',
    '''    if (semantics.feed_forward_norm.after) {
        launch_rmsnorm(w.moe_output.data(), common_layer.feed_forward_norm_after,
                       w.moe_output.data(), rows, context.program.hidden,
                       semantics.feed_forward_norm.after->epsilon, w.stream.get());
    }
    if (semantics.residual.multiplier != 1.0f) {
        launch_scale(w.moe_output.data(), rows * context.program.hidden,
                     semantics.residual.multiplier, w.stream.get());
    }''')
file.write_text(text, encoding="utf-8")

# Residency MoE paths use the same FFN ordering.
file = ROOT / "src/backend/cuda/model/residency.cu"
text = file.read_text(encoding="utf-8")
for buffer, count in [
    ("workspace_.moe_output_.data()", "resources_.program_.hidden"),
    ("workspace_.moe_pf_output_.data()", "rows * resources_.program_.hidden"),
]:
    scale = f'''    if (layer_semantics.residual.multiplier != 1.0f) {{\n        launch_scale({buffer}, {count},\n                     layer_semantics.residual.multiplier, stream_.get());\n    }}\n'''
    if scale in text:
        text = text.replace(scale, "", 1)
        marker = f'''    launch_residual_add(workspace_.hidden_.data(), {buffer},''' if "moe_output_" in buffer and "pf" not in buffer else f'''    launch_residual_add(workspace_.prefill_hidden_.data(), {buffer},'''
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos] + scale + text[pos:]
file.write_text(text, encoding="utf-8")

print("post-norm residual execution order canonicalized")
