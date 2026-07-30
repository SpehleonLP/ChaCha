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
