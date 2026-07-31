#include <gtest/gtest.h>
#include "chacha.h"
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

// Regression coverage for the Critical defect found in whole-branch review:
// chacha_filter.cpp used to drop each rotation RawStage independently by its
// own range against the noise threshold. For a genuine 1-DOF hinge whose
// axis is tilted off the principal axes, the correct proper-Euler solution
// is an ordered triple (constant, sweep, constant) -- e.g. chart XYX with
// slot 0 held at -70deg, slot 1 sweeping [-80,-10]deg, slot 2 held at
// +70deg. The two constant stages have zero range and were deleted by the
// old per-stage filter, even though a zero-range stage in the MIDDLE of an
// ordered composition is a load-bearing fixed frame change, not noise. That
// silently corrupted the joint's reported rotation from an exact fit to one
// with ~74 degrees of reconstruction error.
//
// The fix (chacha_filter.cpp) filters a joint's rotation stages as a single
// all-or-nothing group: if ANY rotation stage clears the range threshold,
// every rotation stage that joint produced -- including zero-range ones --
// is kept. Translation/scale stages are unaffected and still filtered per
// axis.

using namespace ChaCha;
using namespace ChaCha::detail;

namespace {

struct Rig {
    std::vector<int>       parents{-1};
    std::vector<glm::quat> rest{glm::quat(1, 0, 0, 0)};
    std::vector<glm::vec3> trans{glm::vec3(0)};
    std::vector<glm::vec3> scale{glm::vec3(1)};
    Skeleton skeleton() const { return Skeleton{parents, rest, trans, scale}; }
};

bool is_rotation_type(StageType t)
{
    return t == StageType::xRotate || t == StageType::yRotate || t == StageType::zRotate;
}

int axis_for_type(StageType t)
{
    switch (t) {
    case StageType::xRotate: return 0;
    case StageType::yRotate: return 1;
    default:                 return 2;
    }
}

float angular_distance(const glm::quat& a, const glm::quat& b)
{
    const glm::quat an = glm::normalize(a);
    const glm::quat bn = glm::normalize(b);
    const glm::quat d  = glm::inverse(an) * bn;
    const glm::vec3 v(d.x, d.y, d.z);
    return 2.0f * std::atan2(glm::length(v), std::fabs(d.w));
}

// Composes a joint's rotation stages (in emitted order) with a specific
// angle chosen per stage.
glm::quat compose_stages(const std::vector<Stage>& rot_stages, const std::vector<float>& angle)
{
    glm::quat q(1, 0, 0, 0);
    for (size_t i = 0; i < rot_stages.size(); ++i)
        q = q * axis_quat(axis_for_type(rot_stages[i].type), angle[i]);
    return q;
}

// General reconstruction check: does NOT assume anything about which chart
// won or how many stages there are. Given the rotation stages ChaCha
// actually emitted for a joint, searches (coarse-to-fine grid, few rounds)
// for the best admissible angle per stage -- respecting each stage's own
// [min_value, max_value] -- and asserts the composed quaternion reproduces
// the target to within `tol_rad`. This is the reconstruction-fidelity check
// called for in the fix plan: nothing previously verified that emitted
// Stage lists can actually reproduce the input motion, as opposed to
// checking Candidate::summary/internal solver state.
void expect_reconstructible(const std::vector<Stage>& all_stages,
                             const glm::quat& target,
                             float tol_rad)
{
    std::vector<Stage> rot;
    for (const auto& s : all_stages)
        if (is_rotation_type(s.type)) rot.push_back(s);
    ASSERT_FALSE(rot.empty()) << "no rotation stages to reconstruct against";

    std::vector<float> best(rot.size());
    for (size_t i = 0; i < rot.size(); ++i) best[i] = rot[i].min_value;

    auto eval = [&](const std::vector<float>& a) {
        return angular_distance(target, compose_stages(rot, a));
    };

    // Initial pass: a full JOINT cartesian grid over every stage's
    // admissible range. With up to 3 stages a modest per-dim step count is
    // cheap and, unlike per-dimension coordinate descent, cannot get stuck
    // in a local minimum caused by the dimensions interacting (composing
    // three chained rotations is not separable). Coordinate descent below
    // then polishes whatever this pass finds.
    {
        const int joint_steps = (rot.size() == 1) ? 400 : (rot.size() == 2 ? 60 : 22);
        std::vector<int> idx(rot.size(), 0);
        std::vector<float> trial(rot.size());
        float best_err_local = std::numeric_limits<float>::infinity();
        std::function<void(size_t)> recurse = [&](size_t d) {
            if (d == rot.size()) {
                const float err = eval(trial);
                if (err < best_err_local) { best_err_local = err; best = trial; }
                return;
            }
            const float lo = rot[d].min_value, hi = rot[d].max_value;
            for (int i = 0; i <= joint_steps; ++i) {
                trial[d] = lo + (hi - lo) * (float(i) / joint_steps);
                recurse(d + 1);
            }
        };
        recurse(0);
    }

    float best_err = eval(best);

    // Coarse-to-fine grid search per dimension (independently refined over
    // several rounds), polishing the joint-grid result above.
    for (int round = 0; round < 8; ++round) {
        for (size_t d = 0; d < rot.size(); ++d) {
            const float lo = rot[d].min_value;
            const float hi = rot[d].max_value;
            const float full_span = std::max(hi - lo, 1e-6f);
            const float center = (round == 0) ? 0.5f * (lo + hi) : best[d];
            const float half   = (round == 0) ? 0.5f * full_span
                                               : std::max(full_span * std::pow(0.25f, (float)round), 1e-7f);
            const int steps = 40;
            for (int i = 0; i <= steps; ++i) {
                float a = center - half + (2.0f * half) * (float(i) / steps);
                a = std::clamp(a, lo, hi);
                std::vector<float> trial = best;
                trial[d] = a;
                const float err = eval(trial);
                if (err < best_err) { best_err = err; best[d] = a; }
            }
        }
    }

    EXPECT_LE(best_err, tol_rad)
        << "best achievable reconstruction error over admissible stage ranges is "
        << glm::degrees(best_err) << " degrees, exceeds tolerance "
        << glm::degrees(tol_rad) << " degrees";
}

std::vector<Articulation> analyze_single_hinge(const glm::vec3& axis,
                                                float lo_deg, float hi_deg, float step_deg)
{
    Rig rig;
    std::vector<float> times;
    std::vector<float> values;
    int n = 0;
    for (float deg = lo_deg; deg <= hi_deg + 1e-4f; deg += step_deg) {
        const glm::quat q = glm::angleAxis(glm::radians(deg), axis);
        times.push_back(n * 0.1f);
        values.insert(values.end(), {q.x, q.y, q.z, q.w});
        ++n;
    }
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.times = times; ch.values = values;
    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"sweep"}};
    return analyze(chans, anims, rig.skeleton());
}

} // namespace

// Test 1: the headline regression. Axis = Z tilted 20 degrees about X,
// swept 10 to 80 degrees so the motion never touches rest (the trigger
// condition for the defect). Measured against the pre-fix solver this
// FAILED: only one stage (yRotate) was emitted with dof_count == 3, a
// stages.size()==1 vs dof_count==3 mismatch and reconstruction error of
// 74.32 degrees.
TEST(OffAxisHinge, TiltedHingeThroughAnalyzeEmitsAllThreeChartStages)
{
    const glm::quat tilt = glm::angleAxis(glm::radians(20.0f), glm::vec3(1, 0, 0));
    const glm::vec3 axis = glm::normalize(tilt * glm::vec3(0, 0, 1));

    auto out = analyze_single_hinge(axis, 10.0f, 80.0f, 5.0f);
    ASSERT_EQ(out.size(), 1u);
    const Articulation& art = out[0];

    ASSERT_EQ(art.stages.size(), 3u)
        << "the correct proper-Euler fit for this joint has 3 stages "
           "(constant, sweep, constant); the noise filter must not delete "
           "the two zero-range constants";
    EXPECT_EQ(art.dof_count, 3);

    // Slot order: constant, sweep, constant. Outer two constants are equal
    // in magnitude and opposite in sign (the tilt angle); middle sweeps the
    // hinge range.
    EXPECT_NEAR(art.stages[0].max_value - art.stages[0].min_value, 0.0f, 1e-3f);
    EXPECT_NEAR(art.stages[2].max_value - art.stages[2].min_value, 0.0f, 1e-3f);
    EXPECT_GT(art.stages[1].max_value - art.stages[1].min_value, glm::radians(60.0f));

    EXPECT_EQ(art.stages[0].type, art.stages[2].type)
        << "proper-Euler charts repeat the same axis in slots 0 and 2 by construction";
    EXPECT_NEAR(art.stages[0].min_value, -art.stages[2].min_value, 1e-3f);
}

// Test 2: general reconstruction assertion. Composes the ACTUAL emitted
// stages (not internal Candidate/summary state) at their best admissible
// angles and checks the input quaternions are reproduced tightly. Pre-fix,
// the single surviving yRotate stage could not get within 74 degrees of the
// input at any admissible angle; post-fix the full 3-stage set reproduces
// the input essentially exactly.
TEST(OffAxisHinge, EmittedStagesReconstructTheInputMotion)
{
    const glm::quat tilt = glm::angleAxis(glm::radians(20.0f), glm::vec3(1, 0, 0));
    const glm::vec3 axis = glm::normalize(tilt * glm::vec3(0, 0, 1));

    auto out = analyze_single_hinge(axis, 10.0f, 80.0f, 5.0f);
    ASSERT_EQ(out.size(), 1u);

    for (int deg = 10; deg <= 80; deg += 5) {
        const glm::quat target = glm::angleAxis(glm::radians((float)deg), axis);
        expect_reconstructible(out[0].stages, target, glm::radians(1.0f));
    }
}

// Safe case: motion that passes through rest. The proper-Euler gimbal
// singularity at identity forces a different (already 3-varying-stage)
// chart to win, so this was never at risk from the filter bug -- but it is
// the "measure zero knife edge" boundary the defect report calls out, so it
// is pinned here to guard against a fix that overreaches.
TEST(OffAxisHinge, HingeThroughRestStillReconstructsCorrectly)
{
    const glm::quat tilt = glm::angleAxis(glm::radians(20.0f), glm::vec3(1, 0, 0));
    const glm::vec3 axis = glm::normalize(tilt * glm::vec3(0, 0, 1));

    auto out = analyze_single_hinge(axis, -40.0f, 40.0f, 5.0f);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FALSE(out[0].stages.empty());

    for (int deg = -40; deg <= 40; deg += 5) {
        const glm::quat target = glm::angleAxis(glm::radians((float)deg), axis);
        expect_reconstructible(out[0].stages, target, glm::radians(1.0f));
    }
}

// Safe case: an on-axis hinge (no tilt) never exercises the chart-repeated-
// stage mechanism at all -- it is a clean 1-DOF fit -- and must keep working
// exactly as before.
TEST(OffAxisHinge, OnAxisHingeStillProducesASingleStage)
{
    auto out = analyze_single_hinge(glm::vec3(0, 0, 1), 10.0f, 80.0f, 5.0f);
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 1u);
    EXPECT_EQ(out[0].stages[0].type, StageType::zRotate);
    EXPECT_EQ(out[0].dof_count, 1);

    for (int deg = 10; deg <= 80; deg += 5) {
        const glm::quat target = glm::angleAxis(glm::radians((float)deg), glm::vec3(0, 0, 1));
        expect_reconstructible(out[0].stages, target, glm::radians(1.0f));
    }
}

// Safe case, and the critical guard against the fix resurrecting locked
// joints: a genuinely static joint (motion is a fixed offset rotation plus
// noise-level jitter, well under the rotation threshold on every chart
// axis) must still produce NO articulation at all. This is the scenario
// the ruling's rejected "range < threshold && |value| < threshold"
// alternative would have broken: that alternative would have kept a
// zero-range stage sitting at a non-zero constant *value* (like the ~5
// degree offsets observed in the real corpus's currently-locked joints),
// resurrecting an articulation for a joint that should stay reported as
// locked. The per-joint, range-only rule implemented here does not look at
// value at all, so it cannot make that mistake.
TEST(OffAxisHinge, GenuinelyStaticJointProducesNoArticulation)
{
    Rig rig;
    const glm::vec3 base_axis = glm::normalize(glm::vec3(0.2f, 0.9f, 0.3f));

    std::vector<float> times;
    std::vector<float> values;
    for (int i = 0; i < 8; ++i) {
        const float jitter_deg = 0.05f * std::sin(i * 1.3f); // noise-level, << threshold
        const glm::quat q = glm::angleAxis(glm::radians(5.0f + jitter_deg), base_axis);
        times.push_back(i * 0.1f);
        values.insert(values.end(), {q.x, q.y, q.z, q.w});
    }
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.times = times; ch.values = values;
    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"idle"}};

    EXPECT_TRUE(analyze(chans, anims, rig.skeleton()).empty())
        << "a joint whose rotation never clears the noise threshold on any "
           "axis must remain reported as locked (no articulation), exactly "
           "as before the filter fix";
}
