#include "celeg/model/inference.hpp"
#include "support/assertions.hpp"

#include <iostream>

int main() {
    celeg::FactSolver solver;
    const auto proposal = solver.solve<int>({
        {8, {}, celeg::ProposalStrength::ExplicitMetadata, "a"},
        {8, {}, celeg::ProposalStrength::ShapeDerived, "b"}});
    CELEG_TEST_CHECK(proposal.value == 8);

    std::cout << "fact_solver_test: ok\n";
}
