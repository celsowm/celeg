#include "celeg/text/protocol_utils.hpp"
#include "support/assertions.hpp"

#include <string>

int main() {
    CELEG_TEST_CHECK(celeg::json_quote("") == "\"\"");
    CELEG_TEST_CHECK(celeg::json_quote("a\n\t\\\"") == "\"a\\n\\t\\\\\\\"\"");
    CELEG_TEST_CHECK(celeg::json_quote("olá") == "\"olá\"");
    CELEG_TEST_CHECK(celeg::remove_tagged_blocks(
        "before<A>x</A>after<A>y</A>", "<A>", "</A>") == "beforeafter");
    CELEG_TEST_CHECK(celeg::remove_tagged_blocks("x<A>", "<A>", "</A>") == "x");
    return 0;
}
