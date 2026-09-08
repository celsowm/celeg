#pragma once

#include "celeg/model/definition.hpp"

#include <array>
#include <cstdint>

namespace celeg {

void cpu_qk_norm_rope(float* data, const float* norm_weight,
                      int heads, int head_dim, int position,
                      const RopePositionSpec& rope, float eps);
void cpu_qk_norm_only(float* data, const float* norm_weight,
                      int heads, int head_dim, float eps);
void cpu_rope(float* data, int heads, int head_dim, int position,
              const RopePositionSpec& rope);
void cpu_qk_norm_rope_mrope(float* data, const float* norm_weight,
                            int heads, int head_dim,
                            const std::array<int32_t, 3>& positions,
                            const std::array<int, 3>& sections,
                            bool interleaved, const RopePositionSpec& rope,
                            float eps);
void cpu_rope_mrope(float* data, int heads, int head_dim,
                    const std::array<int32_t, 3>& positions,
                    const std::array<int, 3>& sections,
                    bool interleaved, const RopePositionSpec& rope);

}
