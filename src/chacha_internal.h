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

struct RawStage {
    StageType type;
    float min_value{};
    float max_value{};
    float initial_value{};
    float max_velocity{};
    float max_acceleration{};

    float range() const { return max_value - min_value; }
};

std::vector<Stage> filter_stages(
    std::span<const RawStage> raw_stages,
    const Options& options
);

void infer_pointing_vectors(
    std::vector<Articulation>& articulations,
    const Skeleton& skeleton);

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

// Reduced-degree-of-freedom candidate solvers.
//
// solve_one_dof finds the angle `a` such that axis_quat(axis, a) is the
// closest single-axis rotation to `q` (closed form: half-angle extracted
// from q's axis/w components). q and -q represent the same rotation, so
// the input is normalized to the w >= 0 representative before extracting
// the half-angle; without that normalization the extracted angle can be
// off by a multiple of 2pi depending on the arbitrary sign of the input
// quaternion.
float solve_one_dof(const glm::quat& q, int axis);

// Angular distance (radians, via the well-conditioned log-map metric --
// see quaternion_angular_distance below) between q and the single-axis rotation
// axis_quat(axis, angle). Large for motion that a single axis cannot
// represent.
float residual_one_dof(const glm::quat& q, int axis, float angle);

// Gauss-Newton refinement of a 2-axis composition axis_quat(axis0, a0) *
// axis_quat(axis1, a1) toward q, seeded from `seed`. Forward-difference
// Jacobian, damped normal equations, at most 24 iterations.
void solve_two_dof(const glm::quat& q, int axis0, int axis1,
                    const float seed[2], float out[2]);

// Angular distance (radians) between q and the 2-axis composition
// axis_quat(axis0, a[0]) * axis_quat(axis1, a[1]).
float residual_two_dof(const glm::quat& q, int axis0, int axis1, const float a[2]);

// Angular distance (radians) between two unit quaternions, via
// 2*atan2(|v|, |w|) of their relative rotation rather than acos(|dot|) --
// well-conditioned near zero error, unlike acos (see chacha_reduced.cpp).
// General-purpose: used by residual_one_dof/residual_two_dof above and by
// evaluate_chart's real (not assumed-zero) round-trip residual.
float quaternion_angular_distance(const glm::quat& a, const glm::quat& b);

// One joint's rest-pose-relative rotation across every animation that
// touches it. Each inner vector of rel_by_animation/times_by_animation is
// one animation's samples for this joint, in matching order.
struct JointMotion {
    std::vector<std::vector<glm::quat>> rel_by_animation;
    std::vector<std::vector<float>>     times_by_animation;
};

// A candidate rotational decomposition for a joint: `dof` axes (1, 2 or 3),
// each with its StageType and axis index in stage[]/axis[], plus the
// CandidateSummary accumulated (unioned) across every animation in a
// JointMotion. `proper` mirrors Chart::proper for dof == 3 candidates and
// is meaningless otherwise.
//
// Only slots [0, dof) are meaningful. Trailing unused slots are set to
// StageType::Invalid / axis -1, an explicit sentinel rather than the
// enum's/int's zero value (StageType::xTranslate / axis 0), so a caller
// that forgets to bound its read by `dof` gets an obviously-wrong sentinel
// instead of a plausible-looking-but-fabricated translation stage.
//
// IMPORTANT for any consumer (Task 10 and downstream): for the 6 of 12
// three-axis charts where `proper == true` (axis[0] == axis[2], e.g.
// ZXZ), stage[0] and stage[2] are the SAME StageType by construction (the
// same physical axis rotated twice, once before and once after the middle
// stage). This is legal AGI_articulations semantics, not a bug: stages
// apply in order of appearance as an ordered sequence after the node's own
// transform, and a proper-Euler decomposition genuinely needs to name the
// same axis twice with two different, independent ranges. Do not key
// Candidate::summary or downstream Stage lists by StageType -- always
// index by slot/occurrence order (see Task 12's `stage_name_for(type,
// occurrence)`, which exists specifically to disambiguate this). See
// Search.ProperEulerWinnerCanRepeatAStageType in chacha_search_test.cpp,
// which pins that this collision is real, observed, and expected rather
// than something to "fix" by changing the slot convention.
struct Candidate {
    StageType        stage[3]{StageType::Invalid, StageType::Invalid, StageType::Invalid};
    int              axis[3]{-1, -1, -1};
    int              dof{0};
    bool             proper{false};
    CandidateSummary summary;
};

// Enumerates every candidate decomposition of a joint's motion (3 one-axis,
// 6 ordered two-axis, 12 three-axis charts), solves and summarises each
// against every animation in `motion`, and returns the best by a
// lexicographic score: fewest DOF first (subject to a fit-residual gate),
// then smallest summed range, then best worst-case conditioning.
Candidate select_candidate(const JointMotion& motion, const Options& options);

} // namespace detail
} // namespace ChaCha

#endif // CHACHA_INTERNAL_H
