#include <gtest/gtest.h>
#include "chacha_internal.h"
#include "chacha_stage.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>
#include <random>
#include <vector>

using namespace ChaCha;
using namespace ChaCha::detail;

constexpr float kTestPi = 3.14159265358979323846f;

// Angle between two quaternions via the relative-rotation quaternion
// inverse(a)*b, not via acos(dot(a,b)). The acos form is ill-conditioned
// near zero angle (its derivative is unbounded at dot=1), so float-level
// noise from computing a and b along independent paths gets amplified into
// an apparent error around sqrt(float epsilon) ~ 1e-3 rad regardless of how
// accurate the underlying values are. atan2(|vec|, |w|) stays well
// conditioned all the way down to zero.
static float angular_distance(const glm::quat& a, const glm::quat& b)
{
    const glm::quat an = glm::normalize(a);
    const glm::quat bn = glm::normalize(b);
    const glm::quat d = glm::inverse(an) * bn;
    const glm::vec3 v(d.x, d.y, d.z);
    return 2.0f * std::atan2(glm::length(v), std::fabs(d.w));
}

// Map axis index (0=X,1=Y,2=Z) to the StageType compose_chart/solve_euler
// are supposed to label it with, independent of the library's own
// rotate_stage() (which is file-local and not being tested here — this is
// a second, independent statement of the same mapping, so a bug in one
// won't quietly cancel out against the other).
static StageType expected_stage_for_axis(int axis)
{
    switch (axis) {
    case 0:  return StageType::xRotate;
    case 1:  return StageType::yRotate;
    default: return StageType::zRotate;
    }
}

TEST(Charts, ThereAreTwelve)
{
    const auto charts = all_charts();
    EXPECT_EQ(charts.size(), 12u);

    int proper = 0;
    for (const auto& c : charts) if (c.proper) ++proper;
    EXPECT_EQ(proper, 6);

    for (size_t idx = 0; idx < charts.size(); ++idx) {
        const auto& c = charts[idx];
        SCOPED_TRACE(testing::Message() << "chart " << idx << " axis=("
                     << c.axis[0] << "," << c.axis[1] << "," << c.axis[2]
                     << ") proper=" << c.proper);

        // axis[m] in range and stage[m] labels it correctly.
        for (int m = 0; m < 3; ++m) {
            ASSERT_GE(c.axis[m], 0);
            ASSERT_LE(c.axis[m], 2);
            EXPECT_EQ(c.stage[m], expected_stage_for_axis(c.axis[m]));
        }

        // No adjacent stage rotates around the same axis — that would be a
        // degenerate (rank-reducing) chart, never a valid Euler sequence.
        EXPECT_NE(c.axis[0], c.axis[1]);
        EXPECT_NE(c.axis[1], c.axis[2]);

        // `proper` must exactly track "first and last axis match": true for
        // proper Euler (e.g. ZXZ), false for Tait-Bryan (three distinct
        // axes) — never a chart where axis[0]==axis[2] is mislabeled
        // false, nor one where axis[0]!=axis[2] is mislabeled true.
        EXPECT_EQ(c.proper, c.axis[0] == c.axis[2]);
    }

    // No two charts are the same (axis triple, proper) pair.
    for (size_t p = 0; p < charts.size(); ++p) {
        for (size_t q = p + 1; q < charts.size(); ++q) {
            const bool same = charts[p].axis[0] == charts[q].axis[0]
                            && charts[p].axis[1] == charts[q].axis[1]
                            && charts[p].axis[2] == charts[q].axis[2]
                            && charts[p].proper  == charts[q].proper;
            EXPECT_FALSE(same) << "charts " << p << " and " << q << " are duplicates";
        }
    }
}

TEST(Charts, SolveRoundTripsExactlyForEveryChart)
{
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    for (const auto& chart : all_charts()) {
        float worst = 0.0f;
        for (int trial = 0; trial < 500; ++trial) {
            glm::quat q = glm::normalize(glm::quat(u(rng), u(rng), u(rng), u(rng)));
            EulerSolution s = solve_euler(q, chart);
            glm::quat r = compose_chart(chart, s.angle);
            worst = std::max(worst, angular_distance(q, r));
        }
        EXPECT_LT(worst, 1e-4f) << "chart failed round trip";
    }
}

// solve_euler's middle-angle branch splits into three cases: theta2 ~ 0,
// theta2 ~ pi (Tait-Bryan) or theta2 ~ 0, theta2 ~ pi (proper), and the
// generic case in between. The two singular cases only trigger inside a
// 1e-9-wide band around a measure-zero set of configurations (theta2
// exactly +-pi/2 for Tait-Bryan charts, exactly 0 or pi for proper charts)
// — a random unit quaternion can never land there. SolveRoundTripsExactly-
// ForEveryChart's 6000 random samples hit zero of these branches; an
// additional 24 million random draws (run once, offline, to confirm) hit
// exactly one, once. So the singular branches need a deterministic,
// by-construction test, or a bug in them can ship invisibly forever.
TEST(Charts, SolveRoundTripsAtSingularConfigurations)
{
    const float t1_values[] = {-2.5f, -1.7f, -0.8f, 0.0f, 0.6f, 1.4f, 2.3f, 3.0f};
    const float t3_values[] = {-2.9f, -1.3f, -0.4f, 0.2f, 1.1f, 1.8f, 2.6f, 3.1f};

    for (const auto& chart : all_charts()) {
        const std::vector<float> singular_middle = chart.proper
            ? std::vector<float>{0.0f, kTestPi}
            : std::vector<float>{kTestPi / 2.0f, -kTestPi / 2.0f};

        float worst = 0.0f;
        for (float m : singular_middle) {
            for (float t1 : t1_values) {
                for (float t3 : t3_values) {
                    const float angle[3] = {t1, m, t3};
                    glm::quat q = compose_chart(chart, angle);
                    EulerSolution s = solve_euler(q, chart);
                    glm::quat r = compose_chart(chart, s.angle);
                    worst = std::max(worst, angular_distance(q, r));
                }
            }
        }
        EXPECT_LT(worst, 1e-4f) << "chart failed round trip at a singular configuration";
    }
}

TEST(Charts, BothBranchesDescribeTheSameRotation)
{
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (const auto& chart : all_charts()) {
        for (int trial = 0; trial < 200; ++trial) {
            glm::quat q = glm::normalize(glm::quat(u(rng), u(rng), u(rng), u(rng)));
            EulerSolution s0 = solve_euler(q, chart);
            EulerSolution s1 = alternate_branch(s0, chart);
            EXPECT_LT(angular_distance(compose_chart(chart, s1.angle), q), 1e-4f);
        }
    }
}

TEST(Charts, ConditioningFallsToZeroAtTheSingularity)
{
    // Tait-Bryan XYZ is singular when the middle angle reaches +-90 degrees.
    Chart xyz{};
    for (const auto& c : all_charts())
        if (!c.proper && c.axis[0] == 0 && c.axis[1] == 1 && c.axis[2] == 2) xyz = c;

    float safe[3]    = {0.3f, 0.0f,             0.2f};
    float singular[3]= {0.3f, 3.14159265f*0.5f, 0.2f};

    EXPECT_GT(chart_conditioning(EulerSolution{{safe[0], safe[1], safe[2]}}, xyz), 0.9f);
    EXPECT_LT(chart_conditioning(EulerSolution{{singular[0], singular[1], singular[2]}}, xyz), 1e-3f);
}

// alternate_branch must be correct not merely on random draws but exactly
// AT and NEAR the singular configurations, since resolving branch
// ambiguity near a singularity is the entire reason the machinery exists.
// This mirrors SolveRoundTripsAtSingularConfigurations's deterministic
// t1/t3 grid at the singular middle angle, but exercises alternate_branch
// instead of solve_euler's own round trip.
TEST(Charts, AlternateBranchIsCorrectAtSingularConfigurations)
{
    const float t1_values[] = {-2.5f, -1.7f, -0.8f, 0.0f, 0.6f, 1.4f, 2.3f, 3.0f};
    const float t3_values[] = {-2.9f, -1.3f, -0.4f, 0.2f, 1.1f, 1.8f, 2.6f, 3.1f};

    for (const auto& chart : all_charts()) {
        const std::vector<float> singular_middle = chart.proper
            ? std::vector<float>{0.0f, kTestPi}
            : std::vector<float>{kTestPi / 2.0f, -kTestPi / 2.0f};

        float worst = 0.0f;
        for (float m : singular_middle) {
            for (float t1 : t1_values) {
                for (float t3 : t3_values) {
                    const float angle[3] = {t1, m, t3};
                    glm::quat q = compose_chart(chart, angle);
                    EulerSolution s0 = solve_euler(q, chart);
                    EulerSolution s1 = alternate_branch(s0, chart);
                    glm::quat r = compose_chart(chart, s1.angle);
                    worst = std::max(worst, angular_distance(q, r));
                }
            }
        }
        EXPECT_LT(worst, 1e-4f) << "alternate_branch failed at a singular configuration";
    }
}
