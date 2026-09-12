#pragma once

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>

namespace celeg::text {

/// Human-readable binary byte count ("512 B", "1.50 MiB"). Unit selection
/// and precision match the previous per-caller copies exactly.
inline std::string format_bytes(std::size_t bytes) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 3) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2)
        << value << ' ' << units[unit];
    return out.str();
}

}
