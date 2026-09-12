#pragma once

namespace celeg {
class ArchitectureCatalog;
}

namespace celeg::architecture_resolution_test {

void run_failure_modes_tests(const celeg::ArchitectureCatalog& catalog);

void run_late_failure_modes_tests(const celeg::ArchitectureCatalog& catalog);

}
