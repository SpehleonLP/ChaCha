#ifndef CHACHA_H
#define CHACHA_H

#include "chacha_types.h"
#include "chacha_stage.h"

namespace ChaCha {

std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton,
    const Options& options = {}
);

} // namespace ChaCha

#endif // CHACHA_H
