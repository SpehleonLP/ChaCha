#ifndef CHACHA_CLEAN_H
#define CHACHA_CLEAN_H

#include "chacha_types.h"
#include <span>
#include <vector>

namespace ChaCha {

struct CleanOptions {
    float tolerance = 1e-4f;              // Max error for redundant keyframe removal
    float frame_time = 0.0f;             // Target frame duration (s). 0 = disabled.
    bool promote_step_to_linear = false;  // Requires frame_time > 0
    bool promote_linear_to_cubic = false; // Requires frame_time > 0
};

struct CleanedChannel {
    int node{-1};
    Property property{Property::Rotation};
    InterpolationType interp{InterpolationType::Linear};
    std::vector<float> times;
    std::vector<float> values;
};

/// Remove redundant keyframes from animation channels.
/// Returns new owned data with reduced keyframe counts.
std::vector<CleanedChannel> clean(
    std::span<const AnimationChannel> channels,
    const CleanOptions& options = {}
);

} // namespace ChaCha

#endif // CHACHA_CLEAN_H
