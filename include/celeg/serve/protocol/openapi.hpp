#pragma once

#include <string>

namespace celeg::serve::protocol {

std::string build_openapi_spec(const std::string& model_name);

}
