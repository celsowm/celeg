from pathlib import Path

path = Path('include/celeg/model/definition.hpp')
text = path.read_text(encoding='utf-8')
old = '''struct YarnRopeScaling {
    double factor = 1.0;
    double attention_factor = 1.0;
    double beta_fast = 0.0;
    double beta_slow = 0.0;
};'''
new = '''struct YarnRopeScaling {
    double factor = 1.0;
    int original_context = 0;
    double attention_factor = 1.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;

    friend bool operator==(const YarnRopeScaling&, const YarnRopeScaling&) = default;
};'''
if old not in text and new not in text:
    raise SystemExit('current YarnRopeScaling definition not found')
if old in text:
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
print('YaRN definition prepared')