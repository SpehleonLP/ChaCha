#ifndef CHACHA_NAMING_H
#define CHACHA_NAMING_H

#include "chacha_stage.h"
#include <string>
#include <string_view>
#include <vector>

namespace ChaCha {

/// AGI requires articulation names to match ^[^\s]+$ and be unique per model.
/// Collapses whitespace to '_' and appends _2, _3, ... on collision. The name
/// chosen is appended to `taken`.
std::string sanitize_articulation_name(std::string_view raw, std::vector<std::string>& taken);

/// AGI requires stage names to be unique within an articulation. Proper-Euler
/// charts repeat an axis, so `occurrence` disambiguates: 0 -> "zRotate",
/// 1 -> "zRotate2". `type` must not be StageType::Invalid -- that sentinel
/// marks unused Candidate slots and must never reach emitted AGI output.
std::string stage_name_for(StageType type, int occurrence);

} // namespace ChaCha

#endif // CHACHA_NAMING_H
