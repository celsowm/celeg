from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src/model/inference.cpp"
text = path.read_text(encoding="utf-8")

old = '''        const LayerSpec& semantic_layer =
            graph.layers[static_cast<size_t>(layer)];
        require(TensorRole::AttentionInputNorm, layer);

        if (const auto* attention_ptr =
'''
new = '''        const LayerSpec& semantic_layer =
            graph.layers[static_cast<size_t>(layer)];
        const auto require_norm = [&](const std::optional<NormSpec>& norm,
                                      TensorRole role) {
            if (norm.has_value() && !norm->weightless()) require(role, layer);
        };
        require_norm(semantic_layer.mixer_norm.before, TensorRole::AttentionInputNorm);
        require_norm(semantic_layer.mixer_norm.after, TensorRole::AttentionPostNorm);

        if (const auto* attention_ptr =
'''
if old not in text:
    if new not in text:
        raise RuntimeError("canonical mixer norm validation anchor not found")
else:
    text = text.replace(old, new, 1)

old = '''            const AttentionSpec& attention = *attention_ptr;
            if (attention.uses_latent_state()) {
'''
new = '''            const AttentionSpec& attention = *attention_ptr;
            require_norm(attention.query_norm, TensorRole::AttentionQueryNorm);
            require_norm(attention.key_norm, TensorRole::AttentionKeyNorm);
            if (attention.uses_latent_state()) {
'''
if old not in text:
    if new not in text:
        raise RuntimeError("Q/K norm validation anchor not found")
else:
    text = text.replace(old, new, 1)

old = '''        if (!std::holds_alternative<MlpBlockSpec>(semantic_layer.mixer) &&
            !std::holds_alternative<std::monostate>(semantic_layer.feed_forward)) {
            require(TensorRole::FfnInputNorm, layer);
            if (const auto* moe_ptr =
'''
new = '''        if (!std::holds_alternative<MlpBlockSpec>(semantic_layer.mixer) &&
            !std::holds_alternative<std::monostate>(semantic_layer.feed_forward)) {
            require_norm(semantic_layer.feed_forward_norm.before, TensorRole::FfnInputNorm);
            require_norm(semantic_layer.feed_forward_norm.after, TensorRole::FfnOutputNorm);
            if (const auto* moe_ptr =
'''
if old not in text:
    if new not in text:
        raise RuntimeError("canonical FFN norm validation anchor not found")
else:
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("canonical binding validation made topology-aware")
