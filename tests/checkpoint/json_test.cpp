#include "celeg/checkpoint/formats/json.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    const auto value = celeg::Json::parse(R"({"name":"LFM","dims":[1024,2560],"ok":true,"unicode":"olá"})");
    CELEG_TEST_CHECK(value["name"].as_string() == "LFM");
    CELEG_TEST_CHECK(value["dims"].as_array()[1].as_i64() == 2560);
    CELEG_TEST_CHECK(value["ok"].as_bool());
    std::cout << "json_test: ok\n";
}
