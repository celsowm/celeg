from pathlib import Path

path = Path("src/backend/cpu/packed/execution.cpp")
text = path.read_text(encoding="utf-8")
old = '''        auto rmsnorm_rows = [&](const float* input, const std::vector<float>& weight,
                                float* output, size_t width) {
            rows_for([&](size_t row) {
                cpu_rmsnorm(input + row * width, weight.data(), output + row * width,
                            width, shared.program.final_norm.epsilon);
            });
        };
        auto rmsnorm_rows_inplace = [&](float* values, const std::vector<float>& weight,
                                        size_t width) {
            rows_for([&](size_t row) {
                cpu_rmsnorm_inplace(values + row * width, weight.data(), width,
                                    shared.program.final_norm.epsilon);
            });
        };'''
new = '''        auto rmsnorm_rows = [&](const float* input, const std::vector<float>& weight,
                                float* output, size_t width, float epsilon) {
            rows_for([&](size_t row) {
                cpu_rmsnorm(input + row * width, weight.data(), output + row * width,
                            width, epsilon);
            });
        };
        auto rmsnorm_rows_inplace = [&](float* values, const std::vector<float>& weight,
                                        size_t width, float epsilon) {
            rows_for([&](size_t row) {
                cpu_rmsnorm_inplace(values + row * width, weight.data(), width, epsilon);
            });
        };'''
if old not in text:
    raise SystemExit("packed RMSNorm helper block not found")
text = text.replace(old, new, 1)
text = text.replace(
    '''                rmsnorm_rows_inplace(workspace_.hidden.data(), common.per_layer_input_norm, hidden);''',
    '''                rmsnorm_rows_inplace(workspace_.hidden.data(), common.per_layer_input_norm, hidden,
                                     input_plan.norm_epsilon);''')
text = text.replace(
    '''                rmsnorm_rows_inplace(workspace_.hidden.data(),
                                     shared.weight_store.final_norm, hidden);''',
    '''                rmsnorm_rows_inplace(workspace_.hidden.data(),
                                     shared.weight_store.final_norm, hidden,
                                     shared.program.final_norm.epsilon);''')
path.write_text(text, encoding="utf-8")
print("packed RMSNorm helpers now carry explicit epsilon")