#include <gtest/gtest.h>
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <random>

using namespace ChaCha::detail;

static float angular_distance(const glm::quat& a, const glm::quat& b)
{
    float d = std::fabs(glm::dot(glm::normalize(a), glm::normalize(b)));
    if (d > 1.0f) d = 1.0f;
    return 2.0f * std::acos(d);
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
