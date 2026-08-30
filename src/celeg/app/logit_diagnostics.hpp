#pragma once

// Post-prefill logit diagnostics, shared by celeg-run and celeg-cpu-run.
//
// Both apps expose the same two flags so a single file can be compared across
// backends: --dump-logits writes the raw float32 vector for
// celeg-compare-logits, and --print-top names the highest-scoring tokens.
// Backend disagreement on the same checkpoint is the only reference available
// for architectures no other engine can load, so the two apps must emit
// byte-identical formats -- hence one implementation rather than one per app.

#include <string>
#include <vector>

namespace celeg::app {

// Writes the logits as raw little-endian float32, the format
// celeg-compare-logits reads.
void dump_logits_file(const std::string& path, const std::vector<float>& logits);

// Prints the `count` highest logits to stderr as "top[i] token=N logit=X".
// Ties break on the lower token id so the ordering is reproducible across
// backends.
void print_top_logits(const std::vector<float>& logits, int count);

}
