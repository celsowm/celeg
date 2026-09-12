#pragma once

namespace celeg {
class ArchitectureCatalog;
}

namespace celeg::automatic_inference_test {

void run_gguf_resolution_tests(const celeg::ArchitectureCatalog& catalog);

}
