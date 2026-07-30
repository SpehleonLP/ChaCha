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

// A continuous, unwrapped angle trajectory for one chart across a sequence
// of frames. `angle[a]` is not wrapped to (-pi, pi]; it accumulates
// principal differences so that e.g. a joint that keeps rotating the same
// direction keeps growing rather than snapping back into the principal
// branch.
struct Trajectory {
    std::vector<float> time;
    std::vector<float> angle[3];   // continuous, unwrapped, in chart stage order
    float worst_conditioning{1.0f};
};

// Signed difference (current - previous) wrapped into [-pi, pi]. At exactly
// d == pi (mod 2pi) this returns -pi rather than +pi, an artifact of round()
// rounding halves away from zero; the two are equivalent modulo 2pi and this
// is harmless here since callers only ever use the value squared.
float principal_difference(float current, float previous);

// Resolves, per frame, which of solve_euler's two branches minimizes the
// total squared step across all three axes from the previous frame (a
// Viterbi/DP shortest path over the two-branch lattice), then unwraps the
// chosen branch's angles into a continuous trajectory.
//
// The per-frame nearest-neighbour cost alone cannot disambiguate everything:
// flipping EVERY frame's branch choice at once (chart-flip via
// alternate_branch, an involution mod 2pi) leaves every transition cost
// identical, so the whole path and its pointwise flip are always two
// equally optimal solutions to the DP, for any input on any chart. After
// reconstruction, resolve_branches breaks that whole-path tie by comparing
// the reconstructed trajectory's summed squared distance from the rest pose
// (all-zero angles) against its flip's, and keeps whichever is closer to
// rest (ties keep the unflipped reconstruction). The single-frame (n == 1)
// case applies the same nearest-rest criterion directly to the two
// candidate branches, since there is no transition history to run the DP
// over. This tie-break can only choose between the two twins; it cannot
// perturb where a genuine, cost-driven mid-path branch switch occurs,
// because both twins share every transition cost.
Trajectory resolve_branches(
    std::span<const glm::quat> rel_rotations,
    std::span<const float>     times,
    const Chart&               chart);

// Shifts each axis by the multiple of 2pi that brings that axis's trajectory
// midpoint closest to reference[axis]. Trajectories from different animations
// of the same joint must be anchored against a common reference before their
// ranges are unioned, or two animations that straddle the wrap boundary from
// opposite sides will union to a spurious ~2pi span.
// Pass an all-zero reference for the first animation of a joint; for each
// subsequent animation pass the midpoint of the running unioned range. An
// all-zero reference reproduces the (-pi, pi] normalisation of a lone
// trajectory's own midpoint.
void anchor_trajectory(Trajectory& t, const float reference[3]);

// A fixed-size summary of a resolved angle trajectory: observed min/max per
// axis, plus velocity and acceleration recovered by resampling onto a
// uniform time grid (so the derivative estimates are independent of the
// original keyframe density). Unit-agnostic: used for rotation (radians),
// translation (metres) and scale (unitless factor) trajectories alike.
struct CandidateSummary {
    bool  valid{false};
    // Per-axis: whether this axis actually had observed samples. An axis
    // with no samples must not participate in union_into's min/max (its
    // zero-initialised min_value/max_value are not a real observed range
    // and must never drag another summary's range toward zero).
    bool  axis_valid[3]{};
    float min_value[3]{};
    float max_value[3]{};
    float max_velocity[3]{};
    float max_acceleration[3]{};
    float max_residual_rad{0.0f};
    float worst_conditioning{1.0f};
};

CandidateSummary summarise(const Trajectory& t, float residual_rad, const Options& options);
void union_into(CandidateSummary& dst, const CandidateSummary& src);

} // namespace detail
} // namespace ChaCha

#endif // CHACHA_INTERNAL_H
