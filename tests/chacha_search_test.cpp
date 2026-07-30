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

// The same cross-animation wrap defect exists in evaluate_two_dof, not just
// evaluate_one_dof, and needs its own coverage: mutating evaluate_chart's
// own anchor reference to all-zero produced ZERO output difference across
// 200 randomised two-animation joint motions plus this exact construction,
// because the min-total-range tie-break in select_candidate silently
// discards whichever chart candidate blew up to a spurious ~2pi range and
// picks a different (unaffected) chart instead -- so the chart path's
// anchoring has no test that can catch a regression there (see the comment
// on evaluate_chart). The two-DOF path IS directly observable, though: this
// motion is a genuine composition of a Z-axis swing with an X-axis swing
// that straddles the +-180 wrap the same way as
// CrossAnimationAnchorCollapsesOppositeSidesOfTheWrap above, so no 1-DOF
// candidate explains it and dof=2 wins outright (fewer DOF beats every
// 3-DOF chart candidate regardless of range). Confirmed by mutation:
// breaking evaluate_two_dof's cross-animation reference to all-zero widens
// the X-axis slot from ~10 degrees to ~370 degrees here.
TEST(Search, TwoDofCrossAnimationAnchorCollapsesOppositeSidesOfTheWrap)
{
    std::vector<glm::quat> a, b;
    for (int i = 0; i <= 10; ++i)
        a.push_back(glm::angleAxis(glm::radians(175.0f + i), glm::vec3(1, 0, 0))
                  * glm::angleAxis(glm::radians(static_cast<float>(i)), glm::vec3(0, 0, 1)));
    for (int i = 0; i <= 10; ++i)
        b.push_back(glm::angleAxis(glm::radians(185.0f - i), glm::vec3(1, 0, 0))
                  * glm::angleAxis(glm::radians(static_cast<float>(i)), glm::vec3(0, 0, 1)));

    Candidate c = select_candidate(motion_from({a, b}), Options{});
    EXPECT_EQ(c.dof, 2);
    for (int s = 0; s < c.dof; ++s) {
        const float width = c.summary.max_value[s] - c.summary.min_value[s];
        EXPECT_LT(width, glm::radians(30.0f)) << "slot " << s << " width " << glm::degrees(width) << " deg";
    }
}

// Finding 4: the cross-animation protocol anchors each animation against
// the PRIOR running union, so the raw final range is only unique up to a
// 2pi shift per axis -- it depends on which animation the caller happened
// to list first. select_candidate normalises this away (see
// normalize_range_to_nearest_zero in chacha_search.cpp) by shifting the
// final range to whichever 2pi multiple brings its midpoint nearest zero,
// so the two orderings below must report the identical range even though
// this specific 10 degree arc is centred exactly on the +-180 antipode --
// the one input where a naive "round to nearest" tie-break would still
// disagree by sign (see round_half_up's doc comment).
TEST(Search, FinalRangeIsIndependentOfAnimationOrder)
{
    std::vector<glm::quat> a, b;
    for (int i = 0; i <= 10; ++i)
        a.push_back(glm::angleAxis(glm::radians(175.0f + i), glm::vec3(1, 0, 0)));
    for (int i = 0; i <= 10; ++i)
        b.push_back(glm::angleAxis(glm::radians(185.0f - i), glm::vec3(1, 0, 0)));

    Candidate forward = select_candidate(motion_from({a, b}), Options{});
    Candidate reverse = select_candidate(motion_from({b, a}), Options{});

    ASSERT_EQ(forward.dof, 1);
    ASSERT_EQ(reverse.dof, 1);
    EXPECT_NEAR(forward.summary.min_value[0], reverse.summary.min_value[0], 1e-4f);
    EXPECT_NEAR(forward.summary.max_value[0], reverse.summary.max_value[0], 1e-4f);
}

// Finding 3: 6 of the 12 charts are proper-Euler (axis[0] == axis[2]), so a
// proper-Euler winner's stage[0] and stage[2] are the SAME StageType by
// construction -- legal AGI_articulations semantics (an ordered sequence,
// not a set), not a bug. Pinned here rather than left to be "discovered"
// later: over 300 random general-motion joints, 92 (31%) produced exactly
// this collision on a real winner. This construction reliably (not just
// probabilistically) produces a proper-Euler winner by literally generating
// the motion through compose_chart on a known proper chart, with distinct
// magnitudes for the repeated axis's two occurrences so the collision is
// visibly a repeated TYPE with two independently-tracked ranges, not a
// coincidental duplicate.
TEST(Search, ProperEulerWinnerCanRepeatAStageType)
{
    const Chart* proper = nullptr;
    for (const auto& ch : all_charts())
        if (ch.proper) { proper = &ch; break; }
    ASSERT_NE(proper, nullptr);

    std::vector<glm::quat> a;
    std::vector<float> times;
    const int n = 40;
    for (int i = 0; i <= n; ++i) {
        const float f = static_cast<float>(i);
        const float angle[3] = {
            glm::radians(f * 1.0f),          // slot 0 (repeated axis, 1st occurrence): 0..40 deg
            glm::radians(f * 0.7f + 5.0f),   // slot 1 (middle, different axis): 5..33 deg
            glm::radians(f * 0.15f),         // slot 2 (repeated axis, 2nd occurrence): 0..6 deg
        };
        a.push_back(compose_chart(*proper, angle));
        times.push_back(f / 30.0f);
    }

    JointMotion m;
    m.rel_by_animation.push_back(a);
    m.times_by_animation.push_back(times);

    Candidate c = select_candidate(m, Options{});
    ASSERT_EQ(c.dof, 3);
    ASSERT_TRUE(c.proper);
    EXPECT_EQ(c.stage[0], c.stage[2]) << "proper-Euler winner should repeat its outer StageType";
    EXPECT_EQ(c.axis[0], c.axis[2]);

    // The two occurrences of the repeated type carry distinct, independently
    // tracked ranges -- this is not one accidentally-duplicated value.
    const float range0 = c.summary.max_value[0] - c.summary.min_value[0];
    const float range2 = c.summary.max_value[2] - c.summary.min_value[2];
    EXPECT_GT(range0, glm::radians(20.0f));
    EXPECT_LT(range2, glm::radians(15.0f));
}
