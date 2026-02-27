#include "chacha_internal.h"
#include <cmath>

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
        float range = raw.range();
        float threshold = threshold_for_type(raw.type, options);

        if (range < threshold) continue;

        Stage stage;
        stage.type = raw.type;
        stage.min_value = raw.min_value;
        stage.max_value = raw.max_value;
        stage.initial_value = raw.initial_value;
        stage.max_velocity = raw.max_velocity;
        stage.max_effort = raw.max_effort;
        result.push_back(stage);
    }

    return result;
}

} // namespace detail
} // namespace ChaCha
