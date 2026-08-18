from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "tests/automatic_inference_test.cpp"
text = path.read_text(encoding="utf-8")
text = text.replace("left.shape == right.shape", "left.expected_shape == right.expected_shape")
path.write_text(text, encoding="utf-8")
print("weight plan identity assertions aligned")
