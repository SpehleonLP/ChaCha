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
