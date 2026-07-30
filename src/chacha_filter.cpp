#include "chacha_internal.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ChaCha {
namespace detail {

static float threshold_for_type(StageType type, const Options& options)
{
    switch (type) {
    case StageType::xRotate:
    case StageType::yRotate:
    case StageType::zRotate:
        return options.rotation_threshold_rad;

    case StageType::xTranslate:
    case StageType::yTranslate:
    case StageType::zTranslate:
        return options.translation_threshold_m;

    case StageType::xScale:
    case StageType::yScale:
    case StageType::zScale:
        return options.scale_threshold;

    case StageType::Invalid:
        // Invalid is the sentinel for an unused Candidate slot; it must
        // never reach filter_stages, let alone survive it into emitted
        // output. Returning 0 here (the old fallthrough default) would be
        // actively wrong: since range() >= 0 always, `range < 0` is false,
        // so a stray Invalid-typed RawStage would sail through the filter
        // unfiltered instead of being caught. Return +inf so range() can
        // never clear the bar, guaranteeing it is always dropped even if
        // it does turn up here as a bug upstream.
        return std::numeric_limits<float>::infinity();
    }
    return 0.0f;
}

std::vector<Stage> filter_stages(
    std::span<const RawStage> raw_stages,
    const Options& options)
{
    std::vector<Stage> result;
    result.reserve(raw_stages.size());

    for (const auto& raw : raw_stages) {
        const float threshold = threshold_for_type(raw.type, options);
        if (raw.range() < threshold) continue;

        Stage stage;
        stage.type             = raw.type;
        stage.min_value        = raw.min_value;
        stage.max_value        = raw.max_value;
        stage.initial_value    = std::clamp(raw.initial_value, raw.min_value, raw.max_value);
        stage.max_velocity     = raw.max_velocity;
        stage.max_acceleration = raw.max_acceleration;
        result.push_back(stage);
    }

    return result;
}

} // namespace detail
} // namespace ChaCha
