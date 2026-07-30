#ifndef CHACHA_INTERNAL_H
#define CHACHA_INTERNAL_H

#include "chacha_types.h"
#include "chacha_stage.h"
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <span>
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

// A chart is an ordered triple of rotation axes. `axis` holds 0=X, 1=Y, 2=Z.
// `proper` is true for proper-Euler charts (axis[0] == axis[2], e.g. ZXZ) and
// false for Tait-Bryan charts (three distinct axes, e.g. XYZ).
struct Chart {
    StageType stage[3];
    int       axis[3];
    bool      proper;
};

struct EulerSolution {
    float angle[3];
};

std::span<const Chart> all_charts();
EulerSolution solve_euler(const glm::quat& q, const Chart& c);
glm::quat     compose_chart(const Chart& c, const float angle[3]);
glm::quat     axis_quat(int axis, float angle_rad);

// The second Euler-angle solution branch describing the same rotation as
// `s` under chart `c` (the classic "flip" ambiguity: e.g. shoulder-flexed
// vs. shoulder-hyperextended-and-rotated-180 read the same net rotation).
EulerSolution alternate_branch(const EulerSolution& s, const Chart& c);

// 1.0 is perfectly conditioned, 0.0 is exactly singular (gimbal lock).
float chart_conditioning(const EulerSolution& s, const Chart& c);

} // namespace detail
} // namespace ChaCha

#endif // CHACHA_INTERNAL_H
