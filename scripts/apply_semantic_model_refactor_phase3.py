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
    "src/model/descriptor/architecture.cpp",
    "            semantic_layer.operator_norm = {numerical_policy.norm_eps,\n                                             descriptor_.operator_norm_kind};\n            semantic_layer.feed_forward_norm = NormSpec{numerical_policy.norm_eps,\n                                                 descriptor_.feed_forward_norm_kind};",
    "            semantic_layer.mixer_norm.before = NormSpec{\n                numerical_policy.norm_eps, descriptor_.operator_norm_kind};\n            semantic_layer.feed_forward_norm.before = NormSpec{\n                numerical_policy.norm_eps, descriptor_.feed_forward_norm_kind};",
)

print("descriptor semantic norm migration staged")
