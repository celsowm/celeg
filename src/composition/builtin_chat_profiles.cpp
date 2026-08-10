#include "celeg/text/chat_template.hpp"

#include "celeg/models/gemma4/chat_template.hpp"
#include "celeg/models/granite/chat_template.hpp"
#include "celeg/models/lfm2/chat_template.hpp"
#include "celeg/models/minicpm5/chat_template.hpp"
#include "celeg/models/nanbeige/chat_template.hpp"
#include "celeg/models/nemotron_h/chat_template.hpp"
#include "celeg/models/qwen35/chat_template.hpp"
#include "celeg/models/muse_glimmer/chat_template.hpp"
#include "celeg/models/smollm3/chat_template.hpp"

namespace celeg {

ChatProfileCatalog make_chat_profile_catalog() {
    ChatProfileCatalog catalog;
    add_lfm2_chat_profile(catalog);
    add_granite_chat_profile(catalog);
    add_gemma4_chat_profile(catalog);
    add_minicpm5_chat_profile(catalog);
    add_nanbeige42_chat_profile(catalog);
    add_smollm3_chat_profile(catalog);
    add_qwen35_chat_profile(catalog);
    add_muse_glimmer_chat_profile(catalog);
    add_nemotron_h_chat_profile(catalog);
    catalog.freeze();
    return catalog;
}

} // namespace celeg
