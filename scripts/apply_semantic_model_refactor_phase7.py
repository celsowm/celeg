from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_optional(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        file.write_text(text.replace(old, new), encoding="utf-8")


replace_optional(
    "src/backend/cuda/model/prefill_batched/non_attention.cpp",
    "semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon : resources_.program_.final_norm.epsilon",
    "semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon : model.resources_.program_.final_norm.epsilon")

for prefix in ["workspace_.", "workspace_.prefill_"]:
    pass

replace_optional(
    "src/backend/cuda/model/residency.cu",
    '''    launch_rmsnorm(workspace_.hidden_.data(), common_layer.feed_forward_norm_before, workspace_.normed_.data(),\n                    1, resources_.program_.hidden, layer_semantics.feed_forward_norm.before->epsilon, stream_.get());''',
    '''    if (layer_semantics.feed_forward_norm.before) {\n        launch_rmsnorm(workspace_.hidden_.data(), common_layer.feed_forward_norm_before,\n                       workspace_.normed_.data(), 1, resources_.program_.hidden,\n                       layer_semantics.feed_forward_norm.before->epsilon, stream_.get());\n    } else {\n        CELEG_CUDA(cudaMemcpyAsync(\n            workspace_.normed_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),\n            cudaMemcpyDeviceToDevice, stream_.get()));\n    }''')
replace_optional(
    "src/backend/cuda/model/residency.cu",
    '''    launch_residual_add(workspace_.hidden_.data(), workspace_.moe_output_.data(),\n                         resources_.program_.hidden, stream_.get());''',
    '''    if (layer_semantics.residual.multiplier != 1.0f) {\n        launch_scale(workspace_.moe_output_.data(), resources_.program_.hidden,\n                     layer_semantics.residual.multiplier, stream_.get());\n    }\n    if (layer_semantics.feed_forward_norm.after) {\n        launch_rmsnorm(workspace_.moe_output_.data(), common_layer.feed_forward_norm_after,\n                       workspace_.moe_output_.data(), 1, resources_.program_.hidden,\n                       layer_semantics.feed_forward_norm.after->epsilon, stream_.get());\n    }\n    launch_residual_add(workspace_.hidden_.data(), workspace_.moe_output_.data(),\n                        resources_.program_.hidden, stream_.get());''')
replace_optional(
    "src/backend/cuda/model/residency.cu",
    '''    launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.feed_forward_norm_before,\n                   workspace_.prefill_normed_.data(), rows, resources_.program_.hidden, layer_semantics.feed_forward_norm.before->epsilon,\n                   stream_.get());''',
    '''    if (layer_semantics.feed_forward_norm.before) {\n        launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.feed_forward_norm_before,\n                       workspace_.prefill_normed_.data(), rows, resources_.program_.hidden,\n                       layer_semantics.feed_forward_norm.before->epsilon, stream_.get());\n    } else {\n        CELEG_CUDA(cudaMemcpyAsync(\n            workspace_.prefill_normed_.data(), workspace_.prefill_hidden_.data(),\n            static_cast<size_t>(rows) * resources_.program_.hidden * sizeof(__nv_bfloat16),\n            cudaMemcpyDeviceToDevice, stream_.get()));\n    }''')
replace_optional(
    "src/backend/cuda/model/residency.cu",
    '''    launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.moe_pf_output_.data(),\n                        rows * resources_.program_.hidden, stream_.get());''',
    '''    if (layer_semantics.residual.multiplier != 1.0f) {\n        launch_scale(workspace_.moe_pf_output_.data(), rows * resources_.program_.hidden,\n                     layer_semantics.residual.multiplier, stream_.get());\n    }\n    if (layer_semantics.feed_forward_norm.after) {\n        launch_rmsnorm(workspace_.moe_pf_output_.data(), common_layer.feed_forward_norm_after,\n                       workspace_.moe_pf_output_.data(), rows, resources_.program_.hidden,\n                       layer_semantics.feed_forward_norm.after->epsilon, stream_.get());\n    }\n    launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.moe_pf_output_.data(),\n                        rows * resources_.program_.hidden, stream_.get());''')

replace_optional(
    "src/backend/cuda/packed/layer_executor.cu",
    '''        if (!reference.options().fused_residuals) {\n            CELEG_CUDA(cudaMemcpyAsync(\n                workspace_.residual.data(), workspace_.hidden.data(),\n                static_cast<size_t>(rows) * workspace_.program_.hidden *\n                    sizeof(__nv_bfloat16),\n                cudaMemcpyDeviceToDevice, workspace_.stream.get()));\n        }\n        launch_rmsnorm(workspace_.hidden.data(), common_layer.mixer_norm_before,\n                       workspace_.normed.data(), rows, workspace_.program_.hidden,\n                       semantics.mixer_norm.before->epsilon,\n                       workspace_.stream.get());''',
    '''        CELEG_CUDA(cudaMemcpyAsync(\n            workspace_.residual.data(), workspace_.hidden.data(),\n            static_cast<size_t>(rows) * workspace_.program_.hidden * sizeof(__nv_bfloat16),\n            cudaMemcpyDeviceToDevice, workspace_.stream.get()));\n        if (semantics.mixer_norm.before) {\n            launch_rmsnorm(workspace_.hidden.data(), common_layer.mixer_norm_before,\n                           workspace_.normed.data(), rows, workspace_.program_.hidden,\n                           semantics.mixer_norm.before->epsilon, workspace_.stream.get());\n        } else {\n            CELEG_CUDA(cudaMemcpyAsync(\n                workspace_.normed.data(), workspace_.hidden.data(),\n                static_cast<size_t>(rows) * workspace_.program_.hidden * sizeof(__nv_bfloat16),\n                cudaMemcpyDeviceToDevice, workspace_.stream.get()));\n        }''')
replace_optional(
    "src/backend/cuda/packed/layer_executor.cu",
    '''        if (!reference.options().fused_residuals) {\n            launch_residual_add(workspace_.hidden.data(), workspace_.residual.data(),\n                                rows * workspace_.program_.hidden,\n                                workspace_.stream.get());\n        }''',
    '''        if (semantics.mixer_norm.after) {\n            launch_rmsnorm(workspace_.hidden.data(), common_layer.mixer_norm_after,\n                           workspace_.hidden.data(), rows, workspace_.program_.hidden,\n                           semantics.mixer_norm.after->epsilon, workspace_.stream.get());\n        }\n        launch_residual_add(workspace_.hidden.data(), workspace_.residual.data(),\n                            rows * workspace_.program_.hidden, workspace_.stream.get());''')

replace_optional(
    "src/backend/cuda/packed/operators.cu",
    "reference.options().fused_residuals ? 1.0f : 0.0f",
    "0.0f")
replace_optional(
    "src/backend/cuda/packed/operators.cu",
    '''    launch_rmsnorm(w.hidden.data(), common_layer.feed_forward_norm_before, w.normed.data(),\n                   rows, context.program.hidden, semantics.feed_forward_norm.before->epsilon, w.stream.get());''',
    '''    if (semantics.feed_forward_norm.before) {\n        launch_rmsnorm(w.hidden.data(), common_layer.feed_forward_norm_before, w.normed.data(),\n                       rows, context.program.hidden, semantics.feed_forward_norm.before->epsilon,\n                       w.stream.get());\n    } else {\n        CELEG_CUDA(cudaMemcpyAsync(\n            w.normed.data(), w.hidden.data(),\n            static_cast<size_t>(rows) * context.program.hidden * sizeof(__nv_bfloat16),\n            cudaMemcpyDeviceToDevice, w.stream.get()));\n    }''')
replace_optional(
    "src/backend/cuda/packed/operators.cu",
    '''    if (reference.options().fused_residuals) {\n        context.linear(w.activated.data(), *dense->w2, w.hidden.data(), rows,\n                       context.program.hidden, intermediate, 1.0f);\n    } else {\n        context.linear(w.activated.data(), *dense->w2, w.mlp_output.data(), rows,\n                       context.program.hidden, intermediate);\n        launch_scale(w.mlp_output.data(), rows * context.program.hidden,\n                     semantics.residual.multiplier, w.stream.get());\n        launch_residual_add(w.hidden.data(), w.mlp_output.data(),\n                            rows * context.program.hidden, w.stream.get());\n    }''',
    '''    context.linear(w.activated.data(), *dense->w2, w.mlp_output.data(), rows,\n                   context.program.hidden, intermediate);\n    launch_scale(w.mlp_output.data(), rows * context.program.hidden,\n                 semantics.residual.multiplier, w.stream.get());\n    if (semantics.feed_forward_norm.after) {\n        launch_rmsnorm(w.mlp_output.data(), common_layer.feed_forward_norm_after,\n                       w.mlp_output.data(), rows, context.program.hidden,\n                       semantics.feed_forward_norm.after->epsilon, w.stream.get());\n    }\n    launch_residual_add(w.hidden.data(), w.mlp_output.data(),\n                        rows * context.program.hidden, w.stream.get());''')
replace_optional(
    "src/backend/cuda/packed/operators.cu",
    '''    launch_finalize_moe_output(w.moe_output_accum.data(), w.moe_output.data(),\n                               rows * context.program.hidden, w.stream.get());\n    launch_residual_add(w.hidden.data(), w.moe_output.data(),\n                        rows * context.program.hidden, w.stream.get());''',
    '''    launch_finalize_moe_output(w.moe_output_accum.data(), w.moe_output.data(),\n                               rows * context.program.hidden, w.stream.get());\n    launch_scale(w.moe_output.data(), rows * context.program.hidden,\n                 semantics.residual.multiplier, w.stream.get());\n    if (semantics.feed_forward_norm.after) {\n        launch_rmsnorm(w.moe_output.data(), common_layer.feed_forward_norm_after,\n                       w.moe_output.data(), rows, context.program.hidden,\n                       semantics.feed_forward_norm.after->epsilon, w.stream.get());\n    }\n    launch_residual_add(w.hidden.data(), w.moe_output.data(),\n                        rows * context.program.hidden, w.stream.get());''')

replace_optional(
    "src/backend/cuda/packed/operators.cu",
    "semantics.mixer_norm.before->epsilon",
    "semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon : reference.program().final_norm.epsilon")

print("CUDA topology parity staged")
