#include "celeg/model/descriptor.hpp"
#include "detail.hpp"

#include "celeg/checkpoint/formats/json.hpp"
#include "celeg/model/graph_builder.hpp"
#include "celeg/model/weight_plan.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace celeg {
namespace {
using descriptor_detail::AttentionVariant;
using descriptor_detail::BindingPattern;
using descriptor_detail::Descriptor;
using descriptor_detail::Field;
using descriptor_detail::ProbeCondition;
using descriptor_detail::parse_role;
using descriptor_detail::parse_bindings;
using descriptor_detail::parse_probe_conditions;
using descriptor_detail::parse_field;
using descriptor_detail::optional_field;
using descriptor_detail::optional_string;
using descriptor_detail::optional_bool;
using descriptor_detail::parse_attention_variant;
using descriptor_detail::parse_activation_kind;
using descriptor_detail::required;
using descriptor_detail::parse_descriptor;
using descriptor_detail::selected_key;
using descriptor_detail::integer_value;
using descriptor_detail::number_value;
using descriptor_detail::scaling_integer_value;
using descriptor_detail::scaling_number_value;
using descriptor_detail::scaling_factor_values;
using descriptor_detail::position_kind_value;
using descriptor_detail::parse_scaling_kind;
using descriptor_detail::parse_state_scalar;
using descriptor_detail::parse_state_granularity;
using descriptor_detail::scaling_kind_value;
using descriptor_detail::integer_values;
using descriptor_detail::attention_pattern_values;
using descriptor_detail::field_integer_values;
using descriptor_detail::mixer_is_convolution;
using descriptor_detail::mixer_schedule_values;
using descriptor_detail::boolean_value;
using descriptor_detail::token_value;
using descriptor_detail::eos_values;
using descriptor_detail::probe_condition;
using descriptor_detail::build_descriptor_graph;


} // namespace

void register_descriptor_architectures(ArchitectureCatalog& catalog,
                                       const std::filesystem::path& directory) {
    const std::filesystem::path file = directory / "dense_transformers.json";
    if (!std::filesystem::exists(file)) {
        throw std::runtime_error("descriptor store is missing: " + file.string());
    }
    const Json root = Json::parse_file(file.string());
    for (const Json& descriptor : required(root, "architectures").as_array()) {
        const std::string descriptor_id = descriptor.is_object() && descriptor.contains("id")
            ? descriptor.at("id").as_string() : "<unknown>";
        try {
            catalog.add(descriptor_detail::make_descriptor_architecture(
                parse_descriptor(descriptor)));
        } catch (const std::exception& error) {
            throw std::invalid_argument("invalid model descriptor " + descriptor_id + ": " +
                                        std::string(error.what()));
        }
    }
}

} // namespace celeg
