#include <gtest/gtest.h>
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
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
