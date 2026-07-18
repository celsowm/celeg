#include "lfm/json.hpp"
#include <cassert>
#include <iostream>

int main() {
    const auto value = lfm::Json::parse(R"({"name":"LFM","dims":[1024,2560],"ok":true,"unicode":"olá"})");
    assert(value["name"].as_string() == "LFM");
    assert(value["dims"].as_array()[1].as_i64() == 2560);
    assert(value["ok"].as_bool());
    std::cout << "json_test: ok\n";
}
