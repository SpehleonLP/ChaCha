#include <gtest/gtest.h>
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <random>

using namespace ChaCha::detail;

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

TEST(Charts, ThereAreTwelve)
{
    EXPECT_EQ(all_charts().size(), 12u);
    int proper = 0;
    for (const auto& c : all_charts()) if (c.proper) ++proper;
    EXPECT_EQ(proper, 6);
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
