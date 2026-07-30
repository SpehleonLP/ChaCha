#ifndef CHACHA_H
#define CHACHA_H

#include "chacha_types.h"
#include "chacha_stage.h"

namespace ChaCha {

/// Deduce joint articulation constraints from animation data.
///
/// INDEX SPACE: `AnimationChannel::node` indexes the same array as
/// `Skeleton::parents`. ChaCha works in glTF *node* space. Skin-joint space is
/// incorrect: AGI articulations are per node, and glTF animation channels
/// target nodes rather than skin joints.
///
/// `scan` empty means "all animations". In that case, if
/// `options.prioritize_rom_animations` is true and any animation name begins
/// with "AGI " (case-insensitive), the scan narrows to those animations and all
/// others are ignored — an artist-authored configuration animation is a direct
/// specification of the intended constraints. A non-empty `scan` is honoured
/// verbatim with no auto-narrowing.
std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    std::span<const Animation>        animations,
    const Skeleton&                   skeleton,
    const Options&                    options = {},
    std::span<const int>              scan = {},
    std::vector<Diagnostic>*          diagnostics = nullptr);

} // namespace ChaCha

#endif // CHACHA_H
