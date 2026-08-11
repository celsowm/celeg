#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "celeg/text/chat_template_import.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

int main() {
    celeg::CheckpointMetadata flat;
    flat.values["hidden_size"] = std::int64_t{64};
    flat.values["num_hidden_layers"] = std::int64_t{2};
    flat.values["num_attention_heads"] = std::int64_t{4};
    flat.values["vocab_size"] = std::int64_t{128};

    celeg::CheckpointMetadata component;
    component.values["model_type"] = std::string("invented_identity");
    component.values["architectures"] = std::vector<std::string>{"InventedForCausalLM"};
    component.values["text_config.hidden_size"] = std::int64_t{64};
    component.values["text_config.num_hidden_layers"] = std::int64_t{2};
    component.values["text_config.num_attention_heads"] = std::int64_t{4};
    component.values["text_config.vocab_size"] = std::int64_t{128};

    const auto automatic = celeg::make_automatic_architecture();
    CELEG_TEST_CHECK(automatic->probe(flat).supported);
    CELEG_TEST_CHECK(automatic->probe(component).supported);

    const auto flat_facts = celeg::normalize_model_metadata(flat);
    const auto component_facts = celeg::normalize_model_metadata(component);
    CELEG_TEST_CHECK(flat_facts.hidden_size == component_facts.hidden_size);
    CELEG_TEST_CHECK(flat_facts.layer_count == component_facts.layer_count);
    CELEG_TEST_CHECK(flat_facts.query_heads == component_facts.query_heads);
    CELEG_TEST_CHECK(flat_facts.vocab_size == component_facts.vocab_size);

    const celeg::ChatTemplateProgram program = celeg::import_hf_chat_template(
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}");
    CELEG_TEST_CHECK(!program.fingerprint().empty());
    bool unsupported_template_rejected = false;
    try {
        (void)celeg::import_hf_chat_template("{% macro unsafe() %}{% endmacro %}");
    } catch (const std::invalid_argument&) {
        unsupported_template_rejected = true;
    }
    CELEG_TEST_CHECK(unsupported_template_rejected);

    std::cout << "semantic_identity_test: ok\n";
}
