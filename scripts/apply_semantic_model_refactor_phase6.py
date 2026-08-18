from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_optional(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old in text:
        file.write_text(text.replace(old, new), encoding="utf-8")


replace_optional(
    "src/backend/cuda/model/decode.cpp",
    '''    if (!resources_.options_.fused_residuals || common_layer.mixer_norm_after) {\n        CELEG_CUDA(cudaMemcpyAsync(\n            workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),\n            cudaMemcpyDeviceToDevice, stream_.get()));\n    }\n    launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_before,\n                   workspace_.normed_.data(), 1, resources_.program_.hidden,\n                   semantics.mixer_norm.before->epsilon, stream_.get());\n\n    run_token_mixer(layer, common_layer, semantics, layer_index, kv);\n\n    if (common_layer.mixer_norm_after) {\n        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,\n                       workspace_.hidden_.data(), 1, resources_.program_.hidden,\n                       semantics.mixer_norm.after->epsilon, stream_.get());\n    }\n    if (!resources_.options_.fused_residuals || common_layer.mixer_norm_after ||\n        std::holds_alternative<std::monostate>(semantics.feed_forward)) {\n        launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),\n                            resources_.program_.hidden, stream_.get());\n    }''',
    '''    CELEG_CUDA(cudaMemcpyAsync(\n        workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),\n        cudaMemcpyDeviceToDevice, stream_.get()));\n    if (semantics.mixer_norm.before) {\n        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_before,\n                       workspace_.normed_.data(), 1, resources_.program_.hidden,\n                       semantics.mixer_norm.before->epsilon, stream_.get());\n    } else {\n        CELEG_CUDA(cudaMemcpyAsync(\n            workspace_.normed_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),\n            cudaMemcpyDeviceToDevice, stream_.get()));\n    }\n\n    run_token_mixer(layer, common_layer, semantics, layer_index, kv);\n\n    if (semantics.mixer_norm.after) {\n        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,\n                       workspace_.hidden_.data(), 1, resources_.program_.hidden,\n                       semantics.mixer_norm.after->epsilon, stream_.get());\n    }\n    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),\n                        resources_.program_.hidden, stream_.get());''')

for path in [
    "src/backend/cuda/model/decode.cpp",
    "src/backend/cuda/model/attention_execution.cu",
]:
    replace_optional(
        path,
        '''resources_.options_.fused_residuals && !common_layer.mixer_norm_after &&\n               std::holds_alternative<std::monostate>(semantics.feed_forward) ? 0.0f : 1.0f''',
        "0.0f")
    replace_optional(
        path,
        '''resources_.options_.fused_residuals && !common_layer.mixer_norm_after ? 1.0f : 0.0f''',
        "0.0f")

replace_optional(
    "src/backend/cuda/model/decode.cpp",
    "resources_.options_.fused_residuals ? 1.0f : 0.0f",
    "0.0f")
replace_optional(
    "src/backend/cuda/model/non_attention_execution.cu",
    "resources_.options_.fused_residuals ? 1.0f : 0.0f",
    "0.0f")
replace_optional(
    "src/backend/cuda/model/prefill_batched/non_attention.cpp",
    "model.resources_.options_.fused_residuals ? 1.0f : 0.0f",
    "0.0f")

replace_optional(
    "src/backend/cuda/model/execution.cu",
    '''        if (!resources_.options_.fused_residuals) {\n            CELEG_CUDA(cudaMemcpyAsync(\n                workspace_.residual_.data(), workspace_.hidden_.data(),\n                workspace_.hidden_.bytes(), cudaMemcpyDeviceToDevice,\n                stream_.get()));\n        }\n        decode_phase_profile().begin(stream_.get());\n        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_before,\n                       workspace_.normed_.data(), 1, resources_.program_.hidden,\n                       semantics.mixer_norm.before->epsilon, stream_.get());\n        decode_phase_profile().end(DecodePhase::Norm, stream_.get());''',
    '''        CELEG_CUDA(cudaMemcpyAsync(\n            workspace_.residual_.data(), workspace_.hidden_.data(),\n            workspace_.hidden_.bytes(), cudaMemcpyDeviceToDevice,\n            stream_.get()));\n        decode_phase_profile().begin(stream_.get());\n        if (semantics.mixer_norm.before) {\n            launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_before,\n                           workspace_.normed_.data(), 1, resources_.program_.hidden,\n                           semantics.mixer_norm.before->epsilon, stream_.get());\n        } else {\n            CELEG_CUDA(cudaMemcpyAsync(\n                workspace_.normed_.data(), workspace_.hidden_.data(),\n                workspace_.hidden_.bytes(), cudaMemcpyDeviceToDevice, stream_.get()));\n        }\n        decode_phase_profile().end(DecodePhase::Norm, stream_.get());''')
replace_optional(
    "src/backend/cuda/model/execution.cu",
    '''        if (!resources_.options_.fused_residuals || common_layer.mixer_norm_after ||\n            std::holds_alternative<std::monostate>(semantics.feed_forward)) {\n            decode_phase_profile().begin(stream_.get());\n            launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),\n                                resources_.program_.hidden, stream_.get());\n            decode_phase_profile().end(DecodePhase::Other, stream_.get());\n        }''',
    '''        decode_phase_profile().begin(stream_.get());\n        launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),\n                            resources_.program_.hidden, stream_.get());\n        decode_phase_profile().end(DecodePhase::Other, stream_.get());''')

replace_optional(
    "src/backend/cuda/model/prefill_batched.cpp",
    '''    if (!model.resources_.options_.fused_residuals ||\n        common_layer.mixer_norm_after) {\n        CELEG_CUDA(cudaMemcpyAsync(\n            workspace.prefill_residual_.data(),\n            workspace.prefill_hidden_.data(),\n            workspace.prefill_hidden_.bytes(),\n            cudaMemcpyDeviceToDevice, model.stream_.get()));\n    }\n\n    prof.begin(model.stream_.get());\n    launch_rmsnorm(\n        workspace.prefill_hidden_.data(), common_layer.mixer_norm_before,\n        workspace.prefill_normed_.data(),\n        rows, hidden, semantics.mixer_norm.before->epsilon, model.stream_.get());\n    prof.end(PrefillPhase::Norm, model.stream_.get());''',
    '''    CELEG_CUDA(cudaMemcpyAsync(\n        workspace.prefill_residual_.data(), workspace.prefill_hidden_.data(),\n        static_cast<size_t>(rows) * hidden * sizeof(__nv_bfloat16),\n        cudaMemcpyDeviceToDevice, model.stream_.get()));\n\n    prof.begin(model.stream_.get());\n    if (semantics.mixer_norm.before) {\n        launch_rmsnorm(\n            workspace.prefill_hidden_.data(), common_layer.mixer_norm_before,\n            workspace.prefill_normed_.data(), rows, hidden,\n            semantics.mixer_norm.before->epsilon, model.stream_.get());\n    } else {\n        CELEG_CUDA(cudaMemcpyAsync(\n            workspace.prefill_normed_.data(), workspace.prefill_hidden_.data(),\n            static_cast<size_t>(rows) * hidden * sizeof(__nv_bfloat16),\n            cudaMemcpyDeviceToDevice, model.stream_.get()));\n    }\n    prof.end(PrefillPhase::Norm, model.stream_.get());''')
replace_optional(
    "src/backend/cuda/model/prefill_batched.cpp",
    '''    if (!model.resources_.options_.fused_residuals ||\n        common_layer.mixer_norm_after ||\n        std::holds_alternative<std::monostate>(semantics.feed_forward)) {\n        prof.begin(model.stream_.get());\n        launch_residual_add(\n            workspace.prefill_hidden_.data(),\n            workspace.prefill_residual_.data(),\n            rows * hidden, model.stream_.get());\n        prof.end(PrefillPhase::Other, model.stream_.get());\n    }''',
    '''    prof.begin(model.stream_.get());\n    launch_residual_add(\n        workspace.prefill_hidden_.data(), workspace.prefill_residual_.data(),\n        rows * hidden, model.stream_.get());\n    prof.end(PrefillPhase::Other, model.stream_.get());''')

replace_optional(
    "src/backend/cuda/model/dense_mlp_execution.cu",
    '''        launch_rmsnorm(workspace_.hidden_.data(), common_layer.feed_forward_norm_before, workspace_.normed_.data(),\n                       1, resources_.program_.hidden,\n                       semantics.feed_forward_norm.before->epsilon,\n                       stream_.get());''',
    '''        if (semantics.feed_forward_norm.before) {\n            launch_rmsnorm(workspace_.hidden_.data(), common_layer.feed_forward_norm_before,\n                           workspace_.normed_.data(), 1, resources_.program_.hidden,\n                           semantics.feed_forward_norm.before->epsilon, stream_.get());\n        } else {\n            CELEG_CUDA(cudaMemcpyAsync(\n                workspace_.normed_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),\n                cudaMemcpyDeviceToDevice, stream_.get()));\n        }''')
replace_optional(
    "src/backend/cuda/model/dense_mlp_execution.cu",
    '''        launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.feed_forward_norm_before,\n                       workspace_.prefill_normed_.data(), rows, resources_.program_.hidden,\n                       semantics.feed_forward_norm.before->epsilon,\n                       stream_.get());''',
    '''        if (semantics.feed_forward_norm.before) {\n            launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.feed_forward_norm_before,\n                           workspace_.prefill_normed_.data(), rows, resources_.program_.hidden,\n                           semantics.feed_forward_norm.before->epsilon, stream_.get());\n        } else {\n            CELEG_CUDA(cudaMemcpyAsync(\n                workspace_.prefill_normed_.data(), workspace_.prefill_hidden_.data(),\n                static_cast<size_t>(rows) * resources_.program_.hidden * sizeof(__nv_bfloat16),\n                cudaMemcpyDeviceToDevice, stream_.get()));\n        }''')

replace_optional(
    "src/backend/cuda/model/attention_execution.cu",
    "            const float qk_norm_epsilon = layout.query_norm->epsilon;",
    '''            const float qk_norm_epsilon = layout.query_norm\n                ? layout.query_norm->epsilon : resources_.program_.final_norm.epsilon;''')

for path in [
    "src/backend/cuda/model/decode.cpp",
    "src/backend/cuda/model/non_attention_execution.cu",
    "src/backend/cuda/model/prefill_batched/non_attention.cpp",
]:
    replace_optional(
        path,
        "semantics.mixer_norm.before->epsilon",
        "semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon : resources_.program_.final_norm.epsilon")
    replace_optional(
        path,
        "semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon : resources_.program_.final_norm.epsilon ? semantics.mixer_norm.before->epsilon : resources_.program_.final_norm.epsilon",
        "semantics.mixer_norm.before ? semantics.mixer_norm.before->epsilon : resources_.program_.final_norm.epsilon")

print("symmetric CUDA norm execution staged")
