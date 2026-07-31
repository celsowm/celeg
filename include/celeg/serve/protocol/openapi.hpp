#pragma once

#include <string>

namespace celeg::serve::protocol {

// Builds the full OpenAPI 3.1 document (as a JSON string) describing the
// celeg-serve HTTP surface, for the /openapi.json endpoint and Swagger UI.
std::string build_openapi_spec(const std::string& model_name);

} // namespace celeg::serve::protocol
