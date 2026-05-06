#ifndef CHACHA_INTERNAL_H
#define CHACHA_INTERNAL_H

#include "chacha_types.h"
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace ChaCha {
namespace detail {

struct Sample {
    float time{};
    float value{};
    float velocity{};
};

struct DofTrack {
    int node{-1};
    StageType type;
    std::vector<Sample> samples;
};

struct RawStage {
    StageType type;
    float min_value{};
    float max_value{};
    float initial_value{};
    float max_velocity{};
    float max_effort{};

    float range() const { return max_value - min_value; }
};

std::vector<DofTrack> extract_dof_tracks(
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton
);

std::vector<RawStage> segment_and_merge(
    std::span<const DofTrack> tracks_for_joint
);

std::vector<Stage> filter_stages(
    std::span<const RawStage> raw_stages,
    const Options& options
);

void optimize_stage_order(
    std::vector<Stage>& stages,
    const std::vector<DofTrack>& rotation_tracks,
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton,
    int node);

struct SwingTwist {
    glm::quat swing;
    glm::quat twist;
};

void infer_pointing_vectors(
    std::vector<Articulation>& articulations,
    const Skeleton& skeleton);

SwingTwist decompose_swing_twist(const glm::quat& q, const glm::vec3& twist_axis);
glm::vec2 swing_to_angles(const glm::quat& swing);
float twist_to_angle(const glm::quat& twist, const glm::vec3& twist_axis);

} // namespace detail
} // namespace ChaCha

#endif // CHACHA_INTERNAL_H
