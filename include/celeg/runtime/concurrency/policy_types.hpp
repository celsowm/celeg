#pragma once

namespace celeg {

enum class AttentionMode {
    Single,
    Segmented,
    Auto,
};

inline bool select_segmented_attention(AttentionMode mode,
                                       int position,
                                       int auto_threshold) {
    switch (mode) {
        case AttentionMode::Single:
            return false;
        case AttentionMode::Segmented:
            return true;
        case AttentionMode::Auto:
            return position >= auto_threshold;
    }
    return false;
}

} // namespace celeg
