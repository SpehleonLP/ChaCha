#include "chacha_internal.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace ChaCha {
namespace detail {

std::vector<RawStage> segment_and_merge(
    std::span<const DofTrack> tracks_for_joint)
{
    std::map<StageType, RawStage> merged;

    for (const auto& track : tracks_for_joint) {
        auto it = merged.find(track.type);
        if (it == merged.end()) {
            RawStage stage;
            stage.type = track.type;
            stage.min_value = 0.0f;
            stage.max_value = 0.0f;
            stage.initial_value = 0.0f;
            stage.max_velocity = 0.0f;
            stage.max_effort = 0.0f;
            merged.emplace(track.type, stage);
            it = merged.find(track.type);
        }

        RawStage& stage = it->second;

        for (const auto& sample : track.samples) {
            stage.min_value = std::min(stage.min_value, sample.value);
            stage.max_value = std::max(stage.max_value, sample.value);
            stage.max_velocity = std::max(stage.max_velocity, std::abs(sample.velocity));
        }
    }

    std::vector<RawStage> result;
    result.reserve(merged.size());
    for (auto& [type, stage] : merged) {
        result.push_back(std::move(stage));
    }

    return result;
}

} // namespace detail
} // namespace ChaCha
