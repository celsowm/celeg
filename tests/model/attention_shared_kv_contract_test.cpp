#include "celeg/backend/attention_capabilities.hpp"
#include "support/assertions.hpp"

#include <stdexcept>
#include <utility>

namespace {

celeg::CompiledLayerProgram attention_layer(celeg::AttentionSpec attention) {
    celeg::CompiledAttentionProgram compiled;
    compiled.semantics = std::move(attention);
    celeg::CompiledLayerProgram layer;
    layer.mixer = std::move(compiled);
    return layer;
}

celeg::AttentionSpec ordinary_attention(int query_heads = 2) {
    celeg::AttentionSpec attention;
    attention.query_heads = query_heads;
    attention.key_value_heads = 1;
    attention.head_dim = 4;
    attention.position = celeg::NoPositionEncodingSpec{};
    return attention;
}

template <typename Mutator>
bool rejects(Mutator mutate) {
    celeg::CompiledModelProgram program;
    celeg::AttentionSpec publisher = ordinary_attention();
    publisher.kv_sharing = celeg::SharedKvPublisher{7};
    celeg::AttentionSpec consumer = ordinary_attention(4);
    consumer.kv_sharing = celeg::SharedKvConsumer{7};
    program.layers = {
        attention_layer(std::move(publisher)),
        attention_layer(std::move(consumer)),
    };
    mutate(program);
    try {
        celeg::validate_shared_attention_contracts(program);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    CELEG_TEST_CHECK(!rejects([](auto&) {}));

    CELEG_TEST_CHECK(rejects([](auto& program) {
        auto& consumer = std::get<celeg::CompiledAttentionProgram>(
            program.layers[1].mixer).semantics;
        consumer.kv_sharing = celeg::SharedKvConsumer{8};
    }));

    CELEG_TEST_CHECK(rejects([](auto& program) {
        std::swap(program.layers[0], program.layers[1]);
    }));

    CELEG_TEST_CHECK(rejects([](auto& program) {
        auto duplicate = std::get<celeg::CompiledAttentionProgram>(
            program.layers[0].mixer).semantics;
        program.layers.insert(program.layers.begin() + 1,
                              attention_layer(std::move(duplicate)));
    }));

    CELEG_TEST_CHECK(rejects([](auto& program) {
        auto& consumer = std::get<celeg::CompiledAttentionProgram>(
            program.layers[1].mixer).semantics;
        consumer.key_value_heads = 2;
    }));

    CELEG_TEST_CHECK(rejects([](auto& program) {
        auto& consumer = std::get<celeg::CompiledAttentionProgram>(
            program.layers[1].mixer).semantics;
        consumer.head_dim = 8;
    }));

    CELEG_TEST_CHECK(rejects([](auto& program) {
        auto& consumer = std::get<celeg::CompiledAttentionProgram>(
            program.layers[1].mixer).semantics;
        auto& storage = std::get<celeg::OrdinaryKvStateSpec>(consumer.state).storage;
        storage.value = celeg::StateScalarType::FP32;
    }));

    CELEG_TEST_CHECK(rejects([](auto& program) {
        auto& consumer = std::get<celeg::CompiledAttentionProgram>(
            program.layers[1].mixer).semantics;
        consumer.state = celeg::LatentAttentionStateSpec{4, 0, 4, false};
    }));

    return 0;
}
