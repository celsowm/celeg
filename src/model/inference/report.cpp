#include "celeg/model/inference.hpp"

namespace celeg {

ResolutionReport explain_resolution(const CheckpointView& checkpoint) {
    ResolutionReport report;
    report.resolution_mode = "automatic";
    try {
        const InferenceInput input = build_inference_input(checkpoint);
        report.metadata = input.metadata;
        report.chat_template_source = checkpoint.metadata.string_or(
            "chat_template", checkpoint.metadata.string_or("tokenizer.chat_template", {}));
        report.chat_program_fingerprint = report.chat_template_source;
        report.tool_protocol_source = checkpoint.metadata.string_or("tool_protocol", {});
        report.vision_pipeline_source = checkpoint.metadata.string_or("vision_pipeline", {});
        const CanonicalModelFacts facts = infer_canonical_model_facts(input);
        report.accepted = facts.evidence;
        for (const auto& binding : facts.bindings.values) {
            report.accepted.insert(report.accepted.end(), binding.evidence.begin(),
                                   binding.evidence.end());
        }
        report.canonical_fingerprint = facts.fingerprint();
    } catch (const ResolutionError& error) {
        report.failures.push_back(error.what());
        report.rejected = error.evidence();
        if (error.kind() == ResolutionFailureKind::AmbiguousTensorBinding) {
            report.ambiguities.push_back(error.what());
        }
    }
    return report;
}

} // namespace celeg
