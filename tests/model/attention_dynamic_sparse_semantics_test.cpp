#include "celeg/attention/dynamic_sparse_semantics.hpp"
#include "support/assertions.hpp"

namespace {

void content_ranking() {
    using namespace celeg::attention_semantics;
    int blocks[4];
    float selected_scores[4];

    const float scores[] = {1.0f, 4.0f, 2.0f};
    dynamic_sparse_select_top_k(scores, 3, 1, blocks, selected_scores);
    CELEG_TEST_CHECK(blocks[0] == 1);
    CELEG_TEST_CHECK(selected_scores[0] == 4.0f);

    dynamic_sparse_select_top_k(scores, 3, 2, blocks, selected_scores);
    CELEG_TEST_CHECK(blocks[0] == 1);
    CELEG_TEST_CHECK(blocks[1] == 2);
}

void deterministic_ties() {
    using namespace celeg::attention_semantics;
    int blocks[3];
    float selected_scores[3];
    const float tied[] = {3.0f, 3.0f, 2.0f};
    dynamic_sparse_select_top_k(tied, 3, 2, blocks, selected_scores);
    CELEG_TEST_CHECK(blocks[0] == 0);
    CELEG_TEST_CHECK(blocks[1] == 1);
}

void negative_scores_and_capacity() {
    using namespace celeg::attention_semantics;
    int blocks[4];
    float selected_scores[4];
    const float scores[] = {-4.0f, -2.0f, -3.0f};
    dynamic_sparse_select_top_k(scores, 3, 4, blocks, selected_scores);
    CELEG_TEST_CHECK(blocks[0] == 1);
    CELEG_TEST_CHECK(blocks[1] == 2);
    CELEG_TEST_CHECK(blocks[2] == 0);
    CELEG_TEST_CHECK(blocks[3] == -1);
    CELEG_TEST_CHECK(dynamic_sparse_selected_block(2, blocks, 4));
    CELEG_TEST_CHECK(!dynamic_sparse_selected_block(3, blocks, 4));
}

void block_geometry() {
    using namespace celeg::attention_semantics;
    CELEG_TEST_CHECK(dynamic_sparse_query_block(5, 2) == 2);
    CELEG_TEST_CHECK(dynamic_sparse_candidate_count(5, 2) == 3);
    CELEG_TEST_CHECK(dynamic_sparse_block_begin(2, 2) == 4);
    CELEG_TEST_CHECK(dynamic_sparse_block_end_exclusive(5, 2, 2) == 6);
    CELEG_TEST_CHECK(dynamic_sparse_block_end_exclusive(4, 2, 2) == 5);
}

}

int main() {
    content_ranking();
    deterministic_ties();
    negative_scores_and_capacity();
    block_geometry();
    return 0;
}
