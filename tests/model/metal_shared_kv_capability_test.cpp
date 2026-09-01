#include "celeg/backend/metal/attention_capabilities.hpp"
#include "support/assertions.hpp"

#include <stdexcept>
#include <utility>

namespace {

celeg::AttentionSpec attention_with_sharing(celeg::KvSharingSpec sharing) {
    celeg::AttentionSpec attention;
    attention.query_heads = 2;
    attention.key_value_heads = 1;
    attention.head_dim = 8;
    attention.kv_sharing = std::move(sharing);
    return attention;
}

celeg::CompiledLayerProgram layer_with(celeg::AttentionSpec attention) {
    celeg::CompiledAttentionProgram compiled;
    compiled.semantics = std::move(attention);
    compiled.execution.kind = celeg::AttentionExecutionKind::Standard;
    celeg::CompiledLayerProgram layer;
    layer.mixer = std::move(compiled);
    return layer;
}

bool rejects(celeg::CompiledModelProgram program) {
    try {
        celeg::validate_metal_attention_capabilities(program);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    {
        celeg::CompiledModelProgram program;
        program.layers.push_back(layer_with(
            attention_with_sharing(celeg::SharedKvPublisher{7})));
        CELEG_TEST_CHECK(!rejects(std::move(program)));
    }
    {
        celeg::CompiledModelProgram program;
        program.layers.push_back(layer_with(
            attention_with_sharing(celeg::SharedKvConsumer{7})));
        CELEG_TEST_CHECK(rejects(std::move(program)));
    }
    {
        celeg::CompiledModelProgram program;
        program.layers.push_back(layer_with(
            attention_with_sharing(celeg::SharedKvPublisher{7})));
        program.layers.push_back(layer_with(
            attention_with_sharing(celeg::SharedKvPublisher{7})));
        CELEG_TEST_CHECK(rejects(std::move(program)));
    }
    {
        celeg::CompiledModelProgram program;
        program.layers.push_back(layer_with(
            attention_with_sharing(celeg::SharedKvPublisher{7})));
        program.layers.push_back(layer_with(
            attention_with_sharing(celeg::SharedKvConsumer{7})));
        CELEG_TEST_CHECK(rejects(std::move(program)));
    }
    {
        celeg::CompiledModelProgram program;
        auto publisher = attention_with_sharing(celeg::SharedKvPublisher{7});
        auto consumer = attention_with_sharing(celeg::SharedKvConsumer{7});
        consumer.key_value_heads = 2;
        program.layers.push_back(layer_with(std::move(publisher)));
        program.layers.push_back(layer_with(std::move(consumer)));
        CELEG_TEST_CHECK(rejects(std::move(program)));
    }

    return 0;
}
