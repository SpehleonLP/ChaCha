#include <gtest/gtest.h>
#include "chacha_internal.h"
#include "chacha_types.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

using namespace ChaCha;
using namespace ChaCha::detail;

// Mirrors the treefrog layout: X sweep, rest, Z sweep, rest, Y sweep, rest.
// Each phase's signal is 13.4 * sin(f * 2*pi), which passes back through
// rest at the phase's own midpoint (f == 0.5, i.e. i == 20 of 40) as well
// as at its start/end -- so each authored phase is actually observed as
// TWO same-axis sub-runs split by a rest crossing. See the ruling in the
// task brief: re-entering an axis already seen must merge into the
// existing phase (first-encounter order kept), not be rejected as
// interleaving.
static JointMotion configuration_motion()
{
    const glm::vec3 axes[3] = {glm::vec3(1,0,0), glm::vec3(0,0,1), glm::vec3(0,1,0)};
    std::vector<glm::quat> rel;
    std::vector<float> times;
    int frame = 0;
    for (int phase = 0; phase < 3; ++phase) {
        for (int i = 0; i <= 40; ++i) {
            float f = static_cast<float>(i) / 40.0f;
            float deg = 13.4f * std::sin(f * 2.0f * 3.14159265f);
            rel.push_back(glm::angleAxis(glm::radians(deg), axes[phase]));
            times.push_back(frame++ / 24.0f);
        }
    }
    JointMotion m;
    m.rel_by_animation.push_back(std::move(rel));
    m.times_by_animation.push_back(std::move(times));
    return m;
}

TEST(Config, DetectsSequentialSingleAxisPhases)
{
    EXPECT_TRUE(is_configuration_motion(configuration_motion(), Options{}));
}

TEST(Config, RecoversStageOrderAndRange)
{
    Candidate c = solve_configuration(configuration_motion(), Options{});
    ASSERT_EQ(c.dof, 3);
    // Order is the deliverable: a correct-but-wrong-ordered implementation
    // (e.g. sorted by axis index, or by first appearance without honouring
    // interleave-merge) must fail here even though it would report the
    // same three axes and the same ranges.
    EXPECT_EQ(c.stage[0], StageType::xRotate);
    EXPECT_EQ(c.stage[1], StageType::zRotate);
    EXPECT_EQ(c.stage[2], StageType::yRotate);
    for (int s = 0; s < 3; ++s) {
        EXPECT_NEAR(glm::degrees(c.summary.min_value[s]), -13.4f, 1.0f);
        EXPECT_NEAR(glm::degrees(c.summary.max_value[s]),  13.4f, 1.0f);
    }
}

TEST(Config, RejectsOrdinaryMultiAxisMotion)
{
    std::vector<glm::quat> rel; std::vector<float> times;
    for (int i = 0; i <= 60; ++i) {
        float f = static_cast<float>(i);
        rel.push_back(glm::angleAxis(glm::radians(f), glm::vec3(1,0,0))
                    * glm::angleAxis(glm::radians(f * 0.7f), glm::vec3(0,0,1)));
        times.push_back(i / 30.0f);
    }
    JointMotion m;
    m.rel_by_animation.push_back(rel);
    m.times_by_animation.push_back(times);
    EXPECT_FALSE(is_configuration_motion(m, Options{}));
}

TEST(Config, JointHeldAtRestYieldsNoStages)
{
    JointMotion m;
    m.rel_by_animation.push_back({glm::quat(1,0,0,0), glm::quat(1,0,0,0)});
    m.times_by_animation.push_back({0.0f, 5.0f});
    Candidate c = solve_configuration(m, Options{});
    EXPECT_EQ(c.dof, 0);
}

// The brief's own extract_phases closed a phase at every return-to-rest
// and rejected the whole trajectory if any axis appeared in more than one
// phase. A phase that sweeps through rest at its own midpoint (exactly
// what configuration_motion() above does) is split into two same-axis
// phases by that very rule, and its duplicate-axis check then rejected
// it -- so the brief's own supplied test could never have passed against
// its own supplied implementation. This test isolates that single-phase,
// single-axis, rest-crossing-midpoint shape in minimal form: one sweep
// through zero, no other axis involved anywhere, must still be recognised
// as configuration motion with a single merged X-axis phase.
TEST(Config, SameAxisReentryAfterMidPhaseRestCrossingMerges)
{
    std::vector<glm::quat> rel; std::vector<float> times;
    for (int i = 0; i <= 40; ++i) {
        float f = static_cast<float>(i) / 40.0f;
        float deg = 13.4f * std::sin(f * 2.0f * 3.14159265f);
        rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
        times.push_back(i / 24.0f);
    }
    JointMotion m;
    m.rel_by_animation.push_back(rel);
    m.times_by_animation.push_back(times);

    EXPECT_TRUE(is_configuration_motion(m, Options{}));
    Candidate c = solve_configuration(m, Options{});
    ASSERT_EQ(c.dof, 1);
    EXPECT_EQ(c.stage[0], StageType::xRotate);
    EXPECT_NEAR(glm::degrees(c.summary.min_value[0]), -13.4f, 1.0f);
    EXPECT_NEAR(glm::degrees(c.summary.max_value[0]),  13.4f, 1.0f);
}

// Genuine interleaving -- an axis recurring AFTER a different axis has
// intervened (X, then Z, then X again) -- is not a configuration
// animation and must fall back to the search, per the ruling's
// distinction between "re-entry" (merge) and "interleaving" (reject).
TEST(Config, RejectsGenuineInterleaving)
{
    const glm::vec3 x(1, 0, 0), z(0, 0, 1);
    std::vector<glm::quat> rel; std::vector<float> times;
    int frame = 0;
    auto sweep = [&](const glm::vec3& axis) {
        for (int i = 0; i <= 20; ++i) {
            float f = static_cast<float>(i) / 20.0f;
            float deg = 10.0f * std::sin(f * 3.14159265f); // 0 -> 10 -> 0, one-sided
            rel.push_back(glm::angleAxis(glm::radians(deg), axis));
            times.push_back(frame++ / 24.0f);
        }
    };
    sweep(x);
    sweep(z);
    sweep(x);   // X recurs after Z intervened: interleaving, must reject

    JointMotion m;
    m.rel_by_animation.push_back(rel);
    m.times_by_animation.push_back(times);
    EXPECT_FALSE(is_configuration_motion(m, Options{}));
}

namespace {
// A one-sided half-sine sweep (0 -> peak -> 0, never negative) on one
// axis, at a fine enough frame rate that per-frame steps stay well under
// 180 degrees -- the assumption unwrapping via principal_difference (and
// evaluate_one_dof's identical assumption in chacha_search.cpp) relies on.
JointMotion single_axis_sweep(float peak_deg, const glm::vec3& axis, int steps = 120)
{
    std::vector<glm::quat> rel;
    std::vector<float> times;
    for (int i = 0; i <= steps; ++i) {
        const float f   = static_cast<float>(i) / static_cast<float>(steps);
        const float deg = peak_deg * std::sin(f * 3.14159265f);
        rel.push_back(glm::angleAxis(glm::radians(deg), axis));
        times.push_back(static_cast<float>(i) / 60.0f);
    }
    JointMotion m;
    m.rel_by_animation.push_back(std::move(rel));
    m.times_by_animation.push_back(std::move(times));
    return m;
}
} // namespace

// solve_one_dof re-wraps every frame independently into (-pi, pi]: a sweep
// that passes 180 degrees would, without unwrapping, read as snapping to
// the opposite sign rather than continuing to grow. This is precisely the
// intended use case for this fast path -- a deliberately authored
// full-range-of-motion sweep of a large-range joint -- so getting it wrong
// here is a regression relative to the chart search this path bypasses.
TEST(Config, LargeSweepPast180DegreesIsUnwrappedNotWrapped)
{
    Candidate c200 = solve_configuration(single_axis_sweep(200.0f, glm::vec3(1, 0, 0)), Options{});
    ASSERT_EQ(c200.dof, 1);
    EXPECT_NEAR(glm::degrees(c200.summary.min_value[0]), 0.0f, 2.0f);
    EXPECT_NEAR(glm::degrees(c200.summary.max_value[0]), 200.0f, 2.0f);

    Candidate c260 = solve_configuration(single_axis_sweep(260.0f, glm::vec3(1, 0, 0)), Options{});
    ASSERT_EQ(c260.dof, 1);
    EXPECT_NEAR(glm::degrees(c260.summary.min_value[0]), 0.0f, 2.0f);
    EXPECT_NEAR(glm::degrees(c260.summary.max_value[0]), 260.0f, 2.0f);
}

// This path differentiates each phase in isolation (split at every return
// to rest), so it reports a systematically SMALLER peak acceleration than
// select_candidate's chart search would for the same input, which
// differentiates one continuous trajectory across the whole animation and
// so retains the curvature at what would otherwise be a phase boundary.
// Measured on this exact fixture: select_candidate reports roughly
// 13 rad/s^2 here; solve_configuration reports roughly 3.6. Pinned with a
// generous band (not an exact value, which would be fragile to unrelated
// numerical changes) so this divergence cannot silently narrow to zero
// (masking a real behaviour change) or widen further without a test
// noticing.
TEST(Config, AccelerationIsMeasuredPerPhaseNotAcrossPhaseBoundaries)
{
    Candidate c = solve_configuration(configuration_motion(), Options{});
    ASSERT_EQ(c.dof, 3);
    for (int s = 0; s < 3; ++s) {
        EXPECT_GT(c.summary.max_acceleration[s], 1.0f);
        EXPECT_LT(c.summary.max_acceleration[s], 8.0f);
    }
}

// Two separately-authored AGI clips driving the same joint (the real
// shape of e.g. the "AGI Configuration" / "AGI Configuration.001" pair in
// our test corpus): stage order must come from the FIRST animation that
// actually moves, and a second animation's axes must UNION into the
// first's ranges (shared axis) or APPEND after it (new axis) -- not be
// concatenated through one shared run/interleaving state, which is what
// previously made the reported order an artifact of which clip happened
// to sit first in rel_by_animation.
TEST(Config, MultipleAnimationsUnionRangesButFirstAnimationSetsOrder)
{
    JointMotion m;

    // First animation: Z then X, narrow ranges.
    {
        std::vector<glm::quat> rel; std::vector<float> times;
        int frame = 0;
        auto sweep = [&](const glm::vec3& axis, float peak) {
            for (int i = 0; i <= 20; ++i) {
                float f = static_cast<float>(i) / 20.0f;
                float deg = peak * std::sin(f * 3.14159265f);
                rel.push_back(glm::angleAxis(glm::radians(deg), axis));
                times.push_back(frame++ / 24.0f);
            }
        };
        sweep(glm::vec3(0, 0, 1), 5.0f);
        sweep(glm::vec3(1, 0, 0), 10.0f);
        m.rel_by_animation.push_back(rel);
        m.times_by_animation.push_back(times);
    }

    // Second animation: X (shared, wider range) then Y (new axis).
    {
        std::vector<glm::quat> rel; std::vector<float> times;
        int frame = 0;
        auto sweep = [&](const glm::vec3& axis, float peak) {
            for (int i = 0; i <= 20; ++i) {
                float f = static_cast<float>(i) / 20.0f;
                float deg = peak * std::sin(f * 3.14159265f);
                rel.push_back(glm::angleAxis(glm::radians(deg), axis));
                times.push_back(frame++ / 24.0f);
            }
        };
        sweep(glm::vec3(1, 0, 0), 15.0f);   // wider than animation 1's X (10 deg)
        sweep(glm::vec3(0, 1, 0), 3.0f);    // new axis, not seen in animation 1
        m.rel_by_animation.push_back(rel);
        m.times_by_animation.push_back(times);
    }

    EXPECT_TRUE(is_configuration_motion(m, Options{}));
    Candidate c = solve_configuration(m, Options{});
    ASSERT_EQ(c.dof, 3);
    // Order from the FIRST animation: Z, X, then Y appended from the
    // second animation (a genuinely new axis, not a reordering).
    EXPECT_EQ(c.stage[0], StageType::zRotate);
    EXPECT_EQ(c.stage[1], StageType::xRotate);
    EXPECT_EQ(c.stage[2], StageType::yRotate);
    // X's range is the union of both animations' sweeps: [0, 15], not
    // just the first animation's narrower [0, 10].
    EXPECT_NEAR(glm::degrees(c.summary.max_value[1]), 15.0f, 1.0f);
}

// Discriminates cross-animation "concatenation" (the original Finding-1
// bug: all animations fed through one shared run/interleaving state) from
// the correct "first animation sets order, later animations validated and
// merged independently" behaviour. Animation 0 exercises only Z; animation
// 1 independently exercises X, Z, Y in that order (no interleaving WITHIN
// animation 1 itself -- each axis appears once, in sequence). A
// concatenating implementation sees Z (from animation 0) recur in
// animation 1 AFTER X has become the most-recently-added entry, and
// rejects the whole joint as interleaved -- even though nothing here is
// actually ambiguous: Z simply appears in two separate, independently
// clean clips. The correct behaviour merges Z's range into the entry
// animation 0 established and appends X, then Y after it.
TEST(Config, CrossAnimationAxisRecurrenceIsNotInterleaving)
{
    JointMotion m;

    // Animation 0: Z only.
    {
        std::vector<glm::quat> rel; std::vector<float> times;
        for (int i = 0; i <= 20; ++i) {
            float f = static_cast<float>(i) / 20.0f;
            float deg = 6.0f * std::sin(f * 3.14159265f);
            rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(0, 0, 1)));
            times.push_back(i / 24.0f);
        }
        m.rel_by_animation.push_back(rel);
        m.times_by_animation.push_back(times);
    }

    // Animation 1: X, then Z, then Y -- clean in isolation, no axis
    // repeats within this one clip.
    {
        std::vector<glm::quat> rel; std::vector<float> times;
        int frame = 0;
        auto sweep = [&](const glm::vec3& axis, float peak) {
            for (int i = 0; i <= 20; ++i) {
                float f = static_cast<float>(i) / 20.0f;
                float deg = peak * std::sin(f * 3.14159265f);
                rel.push_back(glm::angleAxis(glm::radians(deg), axis));
                times.push_back(frame++ / 24.0f);
            }
        };
        sweep(glm::vec3(1, 0, 0), 9.0f);
        sweep(glm::vec3(0, 0, 1), 7.0f);   // Z recurs (across animations, not within one)
        sweep(glm::vec3(0, 1, 0), 4.0f);
        m.rel_by_animation.push_back(rel);
        m.times_by_animation.push_back(times);
    }

    EXPECT_TRUE(is_configuration_motion(m, Options{}));
    Candidate c = solve_configuration(m, Options{});
    ASSERT_EQ(c.dof, 3);
    EXPECT_EQ(c.stage[0], StageType::zRotate);   // animation 0's order wins
    EXPECT_EQ(c.stage[1], StageType::xRotate);   // appended from animation 1
    EXPECT_EQ(c.stage[2], StageType::yRotate);   // appended from animation 1
    // Z's range unions both animations' sweeps: max(6, 7) = 7 degrees.
    EXPECT_NEAR(glm::degrees(c.summary.max_value[0]), 7.0f, 1.0f);
}

namespace {
// A monotone linear ramp 0 -> peak, followed by a single frame that snaps
// directly back to rest (as opposed to single_axis_sweep's gradual
// half-sine return). Exercises close_run's rest-anchor path from a large,
// actively-growing prev_angle rather than one that has already eased back
// toward 0 -- exactly the case the closing anchor's own derivation must
// get right independent of how the joint got to rest.
JointMotion snap_back_sweep(float peak_deg, const glm::vec3& axis, int steps = 60)
{
    std::vector<glm::quat> rel;
    std::vector<float> times;
    for (int i = 0; i <= steps; ++i) {
        const float f   = static_cast<float>(i) / static_cast<float>(steps);
        const float deg = peak_deg * f;
        rel.push_back(glm::angleAxis(glm::radians(deg), axis));
        times.push_back(static_cast<float>(i) / 60.0f);
    }
    rel.push_back(glm::quat(1, 0, 0, 0));                       // snaps straight back to rest
    times.push_back(static_cast<float>(steps + 1) / 60.0f);
    JointMotion m;
    m.rel_by_animation.push_back(std::move(rel));
    m.times_by_animation.push_back(std::move(times));
    return m;
}
} // namespace

// close_run's rest-anchor sample must close at the SAME (unwrapped) value
// the run began at -- 0 -- never a value derived from the last active
// sample. An earlier version derived the closing anchor via
// principal_difference(0, prev_angle), which steps FORWARD to the nearest
// multiple of 2*pi rather than back to the true rest value whenever
// prev_angle exceeds 180 degrees: a peak-200 sweep that snaps straight
// back to rest reported [0, 360] (a fabricated extra half-turn) instead
// of [0, 200]. Gradual returns (see LargeSweepPast180DegreesIsUnwrappedNotWrapped)
// happen not to hit this, since they ease back through every intermediate
// angle and the true rest-adjacent sample is already near 0 before the
// epsilon check fires; a snap/step return has no such intermediate
// samples, so this is the case that actually exercises the closing
// anchor's own derivation.
TEST(Config, ClosingRestAnchorDoesNotFabricateAFullTurnOnSnapBack)
{
    Candidate c200 = solve_configuration(snap_back_sweep(200.0f, glm::vec3(1, 0, 0)), Options{});
    ASSERT_EQ(c200.dof, 1);
    EXPECT_NEAR(glm::degrees(c200.summary.min_value[0]), 0.0f, 2.0f);
    EXPECT_NEAR(glm::degrees(c200.summary.max_value[0]), 200.0f, 2.0f);

    Candidate c260 = solve_configuration(snap_back_sweep(260.0f, glm::vec3(1, 0, 0)), Options{});
    ASSERT_EQ(c260.dof, 1);
    EXPECT_NEAR(glm::degrees(c260.summary.min_value[0]), 0.0f, 2.0f);
    EXPECT_NEAR(glm::degrees(c260.summary.max_value[0]), 260.0f, 2.0f);
}

namespace {
// A clean, single-axis Y sweep -- unambiguously configuration-shaped on
// its own.
JointMotion clean_y_sweep()
{
    std::vector<glm::quat> rel; std::vector<float> times;
    for (int i = 0; i <= 20; ++i) {
        float f = static_cast<float>(i) / 20.0f;
        float deg = 6.0f * std::sin(f * 3.14159265f);
        rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(0, 1, 0)));
        times.push_back(i / 24.0f);
    }
    JointMotion m;
    m.rel_by_animation.push_back(std::move(rel));
    m.times_by_animation.push_back(std::move(times));
    return m;
}

// A clip with two axes genuinely simultaneous throughout -- fails
// accumulate_animation's single-axis-fit gate on essentially every active
// frame.
std::vector<glm::quat> bad_simultaneous_x_z_frames(std::vector<float>& times)
{
    std::vector<glm::quat> rel;
    for (int i = 0; i <= 20; ++i) {
        const float f = static_cast<float>(i);
        rel.push_back(glm::angleAxis(glm::radians(f), glm::vec3(1, 0, 0))
                    * glm::angleAxis(glm::radians(f), glm::vec3(0, 0, 1)));
        times.push_back(i / 24.0f);
    }
    return rel;
}
} // namespace

// A joint driven by two AGI clips where only one is actually
// configuration-shaped (the scorpion model's real "AGI Configuration" /
// "AGI Configuration.001" pair is exactly this scenario when one clip
// isn't a clean single-axis sweep) must still recover order and range
// from the clip that IS clean -- the bad clip contributes nothing, but
// does not veto the whole joint and force a fall-back to the chart
// search, which would lose the authored order for the one axis that WAS
// deliberately specified. Tested both orderings, since a veto-style bug
// (return false on the first accumulate_animation failure) would behave
// identically regardless of which position the bad clip sits in.
TEST(Config, InvalidAnimationIsSkippedNotVetoedBadClipFirst)
{
    JointMotion m;
    std::vector<float> bad_times;
    m.rel_by_animation.push_back(bad_simultaneous_x_z_frames(bad_times));
    m.times_by_animation.push_back(bad_times);

    JointMotion good = clean_y_sweep();
    m.rel_by_animation.push_back(good.rel_by_animation[0]);
    m.times_by_animation.push_back(good.times_by_animation[0]);

    EXPECT_TRUE(is_configuration_motion(m, Options{}));
    Candidate c = solve_configuration(m, Options{});
    ASSERT_EQ(c.dof, 1);
    EXPECT_EQ(c.stage[0], StageType::yRotate);
    EXPECT_NEAR(glm::degrees(c.summary.max_value[0]), 6.0f, 1.0f);
}

TEST(Config, InvalidAnimationIsSkippedNotVetoedBadClipSecond)
{
    JointMotion m;

    JointMotion good = clean_y_sweep();
    m.rel_by_animation.push_back(good.rel_by_animation[0]);
    m.times_by_animation.push_back(good.times_by_animation[0]);

    std::vector<float> bad_times;
    m.rel_by_animation.push_back(bad_simultaneous_x_z_frames(bad_times));
    m.times_by_animation.push_back(bad_times);

    EXPECT_TRUE(is_configuration_motion(m, Options{}));
    Candidate c = solve_configuration(m, Options{});
    ASSERT_EQ(c.dof, 1);
    EXPECT_EQ(c.stage[0], StageType::yRotate);
    EXPECT_NEAR(glm::degrees(c.summary.max_value[0]), 6.0f, 1.0f);
}
