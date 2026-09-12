#include "automatic_inference/agnes_tests.hpp"
#include "automatic_inference/alias_conflict_tests.hpp"
#include "automatic_inference/gguf_resolution_tests.hpp"
#include "automatic_inference/hf_fixtures_tests.hpp"
#include "automatic_inference/hybrid_gguf_tests.hpp"
#include "automatic_inference/layer_facts_tests.hpp"
#include "automatic_inference/ling_hybrid_tests.hpp"
#include "automatic_inference/qwen35_tests.hpp"
#include "celeg/model/architecture.hpp"

int main() {
    celeg::ArchitectureCatalog catalog;
    catalog.add(celeg::make_automatic_architecture());
    catalog.freeze();

    using namespace celeg::automatic_inference_test;
    run_hf_fixtures_tests(catalog);
    run_gguf_resolution_tests(catalog);
    run_hybrid_gguf_tests(catalog);
    run_alias_conflict_tests();
    run_layer_facts_tests(catalog);
    run_ling_hybrid_tests(catalog);
    run_qwen35_tests(catalog);
    run_agnes_tests(catalog);
    return 0;
}
