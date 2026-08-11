#include "celeg/text/chat_template.hpp"
#include "celeg/text/semantic_chat_templates.hpp"

namespace celeg {

ChatTemplateCatalog make_chat_template_catalog() {
    ChatTemplateCatalog catalog;
    add_delimited_chat_template(catalog);
    add_role_envelope_chat_template(catalog);
    add_turn_chat_template(catalog);
    add_thinking_function_chat_template(catalog);
    add_reasoning_xml_chat_template(catalog);
    add_metadata_thinking_chat_template(catalog);
    add_vision_role_chat_template(catalog);
    add_patch_role_chat_template(catalog);
    add_thinking_role_chat_template(catalog);
    catalog.freeze();
    return catalog;
}

} // namespace celeg
