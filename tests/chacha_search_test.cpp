#include <gtest/gtest.h>
#include "chacha_internal.h"
#include "chacha_types.h"
#include <glm/gtc/quaternion.hpp>

using namespace ChaCha;
using namespace ChaCha::detail;

static JointMotion motion_from(const std::vector<std::vector<glm::quat>>& anims)
{
    JointMotion m;
    for (const auto& a : anims) {
        m.rel_by_animation.push_back(a);
        std::vector<float> t(a.size());
        for (size_t i = 0; i < a.size(); ++i) t[i] = static_cast<float>(i) / 30.0f;
        m.times_by_animation.push_back(std::move(t));
    }
    return m;
}

TEST(Search, PureHingeResolvesToOneDof)
{
    std::vector<glm::quat> a;
    for (int i = 0; i <= 30; ++i)
        a.push_back(glm::angleAxis(glm::radians(i * 2.0f), glm::vec3(1, 0, 0)));
    Candidate c = select_candidate(motion_from({a}), Options{});
    EXPECT_EQ(c.dof, 1);
    EXPECT_EQ(c.stage[0], StageType::xRotate);
}

// The regression that motivated this rework: a pure two-axis swing must not
// manufacture a third rotational degree of freedom.
TEST(Search, TwoAxisSwingProducesNoPhantomThirdDof)
{
    std::vector<glm::quat> a;
    for (int i = 0; i <= 30; ++i)
        a.push_back(glm::angleAxis(glm::radians(45.0f), glm::vec3(1, 0, 0))
                  * glm::angleAxis(glm::radians(i * 3.0f), glm::vec3(0, 0, 1)));
    Candidate c = select_candidate(motion_from({a}), Options{});
    EXPECT_LE(c.dof, 2);
    for (int s = 0; s < c.dof; ++s)
        EXPECT_NE(c.stage[s], StageType::yRotate) << "phantom axial twist";
}

TEST(Search, GeneralMotionUsesThreeDofWithNegligibleResidual)
{
    std::vector<glm::quat> a;
    for (int i = 0; i <= 40; ++i) {
        float f = static_cast<float>(i);
        a.push_back(glm::angleAxis(glm::radians(f * 1.5f), glm::vec3(1, 0, 0))
                  * glm::angleAxis(glm::radians(f * 2.0f), glm::vec3(0, 1, 0))
                  * glm::angleAxis(glm::radians(f * 1.1f), glm::vec3(0, 0, 1)));
    }
    Candidate c = select_candidate(motion_from({a}), Options{});
    EXPECT_EQ(c.dof, 3);
    EXPECT_LT(c.summary.max_residual_rad, 1e-3f);
}

// A single chart must be chosen for the joint as a whole; per-animation chart
// selection would produce angles that cannot be unioned.
TEST(Search, RangeIsUnionedAcrossAnimations)
{
    std::vector<glm::quat> a, b;
    for (int i = 0; i <= 20; ++i)
        a.push_back(glm::angleAxis(glm::radians(i * 1.0f), glm::vec3(1, 0, 0)));
    for (int i = 0; i <= 20; ++i)
        b.push_back(glm::angleAxis(glm::radians(-i * 2.0f), glm::vec3(1, 0, 0)));

    Candidate c = select_candidate(motion_from({a, b}), Options{});
    EXPECT_EQ(c.dof, 1);
    EXPECT_NEAR(glm::degrees(c.summary.min_value[0]), -40.0f, 1.0f);
    EXPECT_NEAR(glm::degrees(c.summary.max_value[0]),  20.0f, 1.0f);
}

// RangeIsUnionedAcrossAnimations above does not actually exercise the
// cross-animation anchoring protocol: verified by hand that it still
// passes even against a select_candidate that anchors every animation
// independently against an all-zero reference (see task-8-report.md). The
// scenario that DOES require the running-union reference: two clips
// describing the identical 10-degree physical arc through the +-180
// wrap, one sampled ascending (175 -> 185, so its own natural midpoint
// lands at +180) and one sampled descending (185 -> 175, so its own
// natural midpoint lands at -180). Anchoring each independently against
// zero sends the two to genuinely opposite branches -- round(+-0.5)
// resolves away from zero in both directions -- unioning to a spurious
// ~370 degree range; anchoring the second against the first's running
// union collapses them back to the true ~10 degree arc.
TEST(Search, CrossAnimationAnchorCollapsesOppositeSidesOfTheWrap)
{
    std::vector<glm::quat> a, b;
    for (int i = 0; i <= 10; ++i)
        a.push_back(glm::angleAxis(glm::radians(175.0f + i), glm::vec3(1, 0, 0)));   // 175 -> 185
    for (int i = 0; i <= 10; ++i)
        b.push_back(glm::angleAxis(glm::radians(185.0f - i), glm::vec3(1, 0, 0)));   // 185 -> 175

    Candidate c = select_candidate(motion_from({a, b}), Options{});
    EXPECT_EQ(c.dof, 1);
    EXPECT_LT(c.summary.max_value[0] - c.summary.min_value[0], glm::radians(30.0f));
}
