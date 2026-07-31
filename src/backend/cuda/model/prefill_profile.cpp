#include "lfm/backend/cuda/phase_profile.hpp"

namespace lfm {

namespace {
PrefillPhaseProfile g_prefill_profile;
}

PrefillPhaseProfile& prefill_phase_profile() { return g_prefill_profile; }

} // namespace lfm
