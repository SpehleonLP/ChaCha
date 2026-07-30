#include <gtest/gtest.h>
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <utility>
#include <vector>

using namespace ChaCha::detail;
static constexpr float kPi = 3.14159265358979323846f;

static Chart tait_xyz()
{
    for (const auto& c : all_charts())
        if (!c.proper && c.axis[0] == 0 && c.axis[1] == 1 && c.axis[2] == 2) return c;
    return all_charts()[0];
}

TEST(DP, PrincipalDifferenceNeverExceedsPi)
{
    EXPECT_NEAR(principal_difference(3.0f, -3.0f), 3.0f - (-3.0f) - 2.0f*kPi, 1e-5f);
    EXPECT_LE(std::fabs(principal_difference(3.0f, -3.0f)), kPi + 1e-5f);
    EXPECT_NEAR(principal_difference(0.1f, 0.0f), 0.1f, 1e-6f);
}

// A joint sweeping smoothly through +-180 degrees must unwrap monotonically
// rather than snapping back into the principal branch.
TEST(DP, UnwrapsMonotonicallyAcrossPi)
{
    std::vector<glm::quat> rel;
    std::vector<float> times;
    for (int i = 0; i <= 40; ++i) {
        float deg = 150.0f + i * 1.5f;           // 150 -> 210 degrees
        times.push_back(i * 0.05f);
        rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
    }
    Trajectory t = resolve_branches(rel, times, tait_xyz());
    for (size_t i = 1; i < t.angle[0].size(); ++i)
        EXPECT_GT(t.angle[0][i], t.angle[0][i - 1]) << "at " << i;

    float span = t.angle[0].back() - t.angle[0].front();
    EXPECT_NEAR(span, glm::radians(60.0f), glm::radians(2.0f));
}

// An aliased fast rotation must resolve to its minimal reading.
TEST(DP, AliasedRotationTakesTheMinimalReading)
{
    std::vector<glm::quat> rel;
    std::vector<float> times;
    for (int i = 0; i < 12; ++i) {
        times.push_back(i / 30.0f);
        rel.push_back(glm::angleAxis(glm::radians(350.0f * i), glm::vec3(1, 0, 0)));
    }
    Trajectory t = resolve_branches(rel, times, tait_xyz());
    float step = t.angle[0][1] - t.angle[0][0];
    EXPECT_NEAR(glm::degrees(step), -10.0f, 1.0f);
}

// The two tests above are both pure single-axis X rotations, for which the
// optimal path never needs to switch Euler-angle branch: a resolver that
// always kept branch 0 would pass both. This test forces the issue by
// holding angle[0] and angle[2] fixed at generic non-zero values while
// sweeping the chart's middle angle straight through its singularity
// (-90 degrees for a Tait-Bryan chart, per chart_conditioning). solve_euler
// necessarily relabels which of its two branches corresponds to "the
// (30, ..., 45) family" on either side of the singular crossing (verified
// directly: at the frame just before the crossing solve_euler's branch 0 is
// (-150, mid, -135); at the frame just after, its branch 0 has jumped to
// (30, mid, 45) -- the family has swapped labels). So the only way to keep
// the reconstructed trajectory continuous is for resolve_branches to switch
// which branch it selects exactly once, at the crossing. A resolver that
// always selects branch 0 reports a ~180 degree discontinuity in angle[0]
// and angle[2] at that frame; this test fails against such a stub (verified
// by temporarily forcing chosen[t]=0 in resolve_branches and re-running).
TEST(DP, SwitchesBranchAcrossChartSingularity)
{
    Chart c = tait_xyz();
    std::vector<glm::quat> rel;
    std::vector<float> times;
    const int N = 41;
    for (int i = 0; i < N; ++i) {
        // -99.75 .. -79.75 degrees: crosses the -90 degree singularity
        // between two sample points without ever landing exactly on it,
        // where the decomposition would be genuinely ill-defined.
        float mid = glm::radians(-99.75f + i * 0.5f);
        float angle[3] = {glm::radians(30.0f), mid, glm::radians(45.0f)};
        rel.push_back(compose_chart(c, angle));
        times.push_back(i * 0.02f);
    }
    Trajectory t = resolve_branches(rel, times, c);
    ASSERT_EQ(t.angle[0].size(), static_cast<size_t>(N));
    for (size_t i = 1; i < t.angle[0].size(); ++i) {
        EXPECT_LT(std::fabs(glm::degrees(t.angle[0][i] - t.angle[0][i - 1])), 5.0f) << "at " << i;
        EXPECT_LT(std::fabs(glm::degrees(t.angle[2][i] - t.angle[2][i - 1])), 5.0f) << "at " << i;
    }
}

// Contract: resolve_branches processes only as many frames as both spans
// actually provide, taking the leading (index 0..n) slice of each -- never
// reading past the end of either span. This test pins that contract with
// `times` shorter than `rel`.
TEST(DP, TruncatesToShorterSpan_TimesShorterThanRel)
{
    Chart c = tait_xyz();
    // Small, well-separated angles with no wraparound and nowhere near the
    // chart's singularity, so branch 0 is unambiguously optimal throughout
    // and each frame's angle[0] equals its input angle exactly.
    std::vector<glm::quat> rel;
    for (float deg : {10.0f, 20.0f, 30.0f, 40.0f, 50.0f})
        rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
    std::vector<float> times = {0.0f, 0.1f, 0.2f};   // shorter than rel (5 frames)

    Trajectory t = resolve_branches(rel, times, c);

    ASSERT_EQ(t.time.size(), times.size());
    ASSERT_EQ(t.angle[0].size(), times.size());
    EXPECT_EQ(t.time, times);   // the retained times are exactly the first 3, unchanged
    for (size_t i = 0; i < t.time.size(); ++i)
        EXPECT_NEAR(glm::degrees(t.angle[0][i]), 10.0f * (i + 1), 1e-3f) << "at " << i;
}

// Same contract, other direction: `rel` shorter than `times`.
TEST(DP, TruncatesToShorterSpan_RelShorterThanTimes)
{
    Chart c = tait_xyz();
    std::vector<glm::quat> rel;
    for (float deg : {10.0f, 20.0f, 30.0f})
        rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
    std::vector<float> times = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f};   // longer than rel (3 frames)

    Trajectory t = resolve_branches(rel, times, c);

    ASSERT_EQ(t.time.size(), rel.size());
    ASSERT_EQ(t.angle[0].size(), rel.size());
    EXPECT_EQ(t.time[0], times[0]);
    EXPECT_EQ(t.time[1], times[1]);
    EXPECT_EQ(t.time[2], times[2]);   // the retained times are exactly the leading 3
    for (size_t i = 0; i < t.time.size(); ++i)
        EXPECT_NEAR(glm::degrees(t.angle[0][i]), 10.0f * (i + 1), 1e-3f) << "at " << i;
}

// Contract: n == 0 yields an empty trajectory, worst_conditioning left at
// its default (best-possible, 1.0), not some computed-from-nothing value.
TEST(DP, EmptyInputYieldsEmptyTrajectory)
{
    std::vector<glm::quat> rel;
    std::vector<float> times;
    Trajectory t = resolve_branches(rel, times, tait_xyz());

    EXPECT_TRUE(t.time.empty());
    for (int a = 0; a < 3; ++a)
        EXPECT_TRUE(t.angle[a].empty());
    EXPECT_FLOAT_EQ(t.worst_conditioning, 1.0f);
}

// Contract: with a single frame there is no history to disambiguate
// branches, so the branch-0 solution passes through unchanged.
TEST(DP, SingleFramePassesThroughBranchZeroUnchanged)
{
    Chart c = tait_xyz();
    float angle[3] = {glm::radians(10.0f), glm::radians(20.0f), glm::radians(30.0f)};
    glm::quat q = compose_chart(c, angle);
    EulerSolution expected = solve_euler(q, c);

    std::vector<glm::quat> rel = {q};
    std::vector<float> times = {0.25f};
    Trajectory t = resolve_branches(rel, times, c);

    ASSERT_EQ(t.time.size(), 1u);
    EXPECT_FLOAT_EQ(t.time[0], 0.25f);
    for (int a = 0; a < 3; ++a) {
        ASSERT_EQ(t.angle[a].size(), 1u);
        EXPECT_NEAR(t.angle[a][0], expected.angle[a], 1e-6f) << "axis " << a;
    }
    EXPECT_FLOAT_EQ(t.worst_conditioning, chart_conditioning(expected, c));
}

// Two animations of one joint that straddle +-180 degrees from opposite sides
// must anchor into a common frame, or their union spans a spurious ~360 degrees.
//
// This builds the two Trajectory objects directly rather than routing them
// through resolve_branches. resolve_branches's Viterbi resolves a *second*,
// independent ambiguity for a pure single-axis rotation: the whole-path cost
// of staying on solve_euler's principal branch is exactly tied (up to float
// rounding) with the whole-path cost of staying on its chart-flip alternate
// (alternate_branch is an exact isometry of the squared-step metric, constant
// offsets cancel in every consecutive difference) -- verified directly: for
// this test's own two sweeps computed by hand, resolve_branches picked
// opposite chart-flip families for the two animations (one landed on
// (angle0, 0, 0), the other on (angle0, 180, 180)), a ~180 degree family
// mismatch that no multiple-of-2pi shift on axis 0 can repair. That is a
// resolve_branches/chart concern (a discrete branch-labelling ambiguity),
// not the continuous 2pi wrap-boundary ambiguity anchor_trajectory exists to
// fix, so it is tested here by constructing the Trajectory inputs directly,
// isolating anchor_trajectory as the unit under test.
TEST(DP, AnchoringKeepsIndependentAnimationsCommensurable)
{
    Trajectory a;
    a.time = {0.0f, 1.0f};
    a.angle[0] = {glm::radians(170.0f), glm::radians(185.0f)};   // crosses 180 upward

    Trajectory b;
    b.time = {0.0f, 1.0f};
    b.angle[0] = {glm::radians(-170.0f), glm::radians(-185.0f)}; // same physical range,
                                                                  // read from the other
                                                                  // side of the wrap

    const float zero_ref[3] = {0.0f, 0.0f, 0.0f};
    anchor_trajectory(a, zero_ref);

    // Anchor b against a's post-anchoring per-axis midpoint, the running
    // reference a caller would carry forward when unioning ranges across
    // animations of the same joint. Axes 1 and 2 are unused (empty) in this
    // synthetic Trajectory, so only reference[0] is meaningful.
    float a_lo = *std::min_element(a.angle[0].begin(), a.angle[0].end());
    float a_hi = *std::max_element(a.angle[0].begin(), a.angle[0].end());
    const float a_ref[3] = {0.5f * (a_lo + a_hi), 0.0f, 0.0f};
    anchor_trajectory(b, a_ref);

    float lo = std::min(*std::min_element(a.angle[0].begin(), a.angle[0].end()),
                        *std::min_element(b.angle[0].begin(), b.angle[0].end()));
    float hi = std::max(*std::max_element(a.angle[0].begin(), a.angle[0].end()),
                        *std::max_element(b.angle[0].begin(), b.angle[0].end()));

    // True excursion is 170..190 degrees, i.e. 20 degrees wide.
    EXPECT_LT(glm::degrees(hi - lo), 40.0f);
}

// With an all-zero reference, anchor_trajectory must reproduce the
// documented (-pi, pi] normalisation of a single trajectory's own midpoint:
// this is the case Tasks 8, 10 and 11 depend on being unchanged from the
// single-animation behaviour.
TEST(DP, ZeroReferenceNormalizesSingleTrajectoryIntoPrincipalRange)
{
    // 0 -> 400 degrees: a single monotonic sweep that crosses the +-180
    // wrap boundary once, so resolve_branches's continuous unwrap keeps
    // growing straight through it (verified directly: front/back land at
    // exactly 0 and 400 degrees, not aliased), leaving a midpoint of 200
    // degrees -- well outside (-pi, pi].
    std::vector<glm::quat> rel;
    std::vector<float> times;
    for (int i = 0; i <= 20; ++i) {
        float f = static_cast<float>(i) / 20.0f;
        float deg = f * 400.0f;
        times.push_back(i * 0.05f);
        rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
    }
    Trajectory t = resolve_branches(rel, times, tait_xyz());

    const float zero_ref[3] = {0.0f, 0.0f, 0.0f};
    anchor_trajectory(t, zero_ref);

    float lo = *std::min_element(t.angle[0].begin(), t.angle[0].end());
    float hi = *std::max_element(t.angle[0].begin(), t.angle[0].end());
    float mid = 0.5f * (lo + hi);

    // Midpoint (200 degrees) shifts by -2pi (360 degrees), landing at
    // -160 degrees, squarely inside (-pi, pi].
    EXPECT_NEAR(glm::degrees(mid), -160.0f, 1.0f);
    EXPECT_GT(mid, -kPi);
    EXPECT_LE(mid, kPi);
}

// Anchoring an already-anchored trajectory against the same reference must
// be a no-op.
TEST(DP, AnchoringIsIdempotent)
{
    // 0 -> 400 degrees, same construction as the zero-reference test above:
    // the first anchor call must actually shift this trajectory by -2pi
    // (its midpoint, 200 degrees, is well outside (-pi, pi]), so the "no
    // change on the second call" assertion below cannot be satisfied
    // vacuously by a no-op anchor_trajectory -- the first call is checked
    // to have moved the data before the second call is checked to leave it
    // alone.
    std::vector<glm::quat> rel;
    std::vector<float> times;
    for (int i = 0; i <= 20; ++i) {
        float f = static_cast<float>(i) / 20.0f;
        float deg = f * 400.0f;
        times.push_back(i * 0.05f);
        rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
    }
    Trajectory t = resolve_branches(rel, times, tait_xyz());
    std::vector<float> before = t.angle[0];

    const float zero_ref[3] = {0.0f, 0.0f, 0.0f};
    anchor_trajectory(t, zero_ref);
    std::vector<float> once = t.angle[0];

    // The first call must have actually moved the data (rules out a no-op
    // anchor_trajectory passing this test vacuously).
    ASSERT_EQ(before.size(), once.size());
    bool changed = false;
    for (size_t i = 0; i < before.size(); ++i)
        if (std::fabs(before[i] - once[i]) > 1e-3f) changed = true;
    EXPECT_TRUE(changed);

    anchor_trajectory(t, zero_ref);

    ASSERT_EQ(once.size(), t.angle[0].size());
    for (size_t i = 0; i < once.size(); ++i)
        EXPECT_FLOAT_EQ(once[i], t.angle[0][i]) << "at " << i;
}
