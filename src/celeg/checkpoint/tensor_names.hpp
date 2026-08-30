#pragma once


#include <string>

namespace celeg {

inline std::string layer_name(int index, const std::string& suffix) {
    return "model.layers." + std::to_string(index) + "." + suffix;
}

}
