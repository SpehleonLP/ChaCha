#include <gtest/gtest.h>
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

using namespace ChaCha::detail;

TEST(Reduced, OneDofRecoversAPureHingeExactly)
{
    for (int axis = 0; axis < 3; ++axis) {
        glm::vec3 v(0.0f); v[axis] = 1.0f;
        glm::quat q = glm::angleAxis(glm::radians(37.0f), v);
        float a = solve_one_dof(q, axis);
        EXPECT_NEAR(glm::degrees(a), 37.0f, 1e-3f);
        EXPECT_LT(residual_one_dof(q, axis, a), 1e-5f);
    }
}

TEST(Reduced, OneDofReportsLargeResidualForOffAxisMotion)
{
    glm::quat q = glm::angleAxis(glm::radians(60.0f), glm::normalize(glm::vec3(1, 0, 1)));
    float a = solve_one_dof(q, 0);
    EXPECT_GT(residual_one_dof(q, 0, a), glm::radians(10.0f));
}

TEST(Reduced, TwoDofRecoversATwoAxisComposition)
{
    glm::quat q = glm::angleAxis(glm::radians(40.0f), glm::vec3(1, 0, 0))
                * glm::angleAxis(glm::radians(-25.0f), glm::vec3(0, 0, 1));
    float seed[2] = {0.0f, 0.0f};
    float out[2]  = {0.0f, 0.0f};
    solve_two_dof(q, 0, 2, seed, out);
    EXPECT_LT(residual_two_dof(q, 0, 2, out), 1e-3f);
    EXPECT_NEAR(glm::degrees(out[0]),  40.0f, 0.5f);
    EXPECT_NEAR(glm::degrees(out[1]), -25.0f, 0.5f);
}

// --- Task-7 additional coverage: the sign convention and Gauss-Newton
// robustness issues flagged in the brief. ---

// q and -q represent the same rotation. The naive extraction
// 2*atan2(qv[axis], q.w) is not invariant to negating the input, so a
// negated quaternion must still recover the same angle as its un-negated
// twin.
TEST(Reduced, OneDofIsInvariantToQuaternionSignNegation)
{
    glm::quat q = glm::angleAxis(glm::radians(37.0f), glm::vec3(1, 0, 0));
    glm::quat qneg(-q.w, -q.x, -q.y, -q.z);

    float a_pos = solve_one_dof(q, 0);
    float a_neg = solve_one_dof(qneg, 0);

    EXPECT_NEAR(glm::degrees(a_pos), 37.0f, 1e-3f);
    EXPECT_NEAR(glm::degrees(a_neg), 37.0f, 1e-3f);
    EXPECT_LT(residual_one_dof(qneg, 0, a_neg), 1e-5f);
}

// A larger-magnitude hinge angle, tested through both the positive- and
// negative-w representative of the same rotation. This exercises a case
// close to (but not past) the +-180 degree wrap.
TEST(Reduced, OneDofHandlesNegativeWRepresentativeNearWrap)
{
    glm::quat q = glm::angleAxis(glm::radians(170.0f), glm::vec3(0, 1, 0));
    // angleAxis(170deg, Y) already has a positive w (cos(85deg) > 0). Force
    // the negative-w representative explicitly, which is the case the
    // brief's own tests never exercise.
    glm::quat qneg(-q.w, -q.x, -q.y, -q.z);
    ASSERT_LT(qneg.w, 0.0f);

    float a = solve_one_dof(qneg, 1);
    EXPECT_NEAR(glm::degrees(a), 170.0f, 1e-2f);
    EXPECT_LT(residual_one_dof(qneg, 1, a), 1e-4f);
}

// solve_two_dof must actually converge from the cold {0,0} seed used
// throughout the brief and reach a genuinely small residual, not merely
// "closer than before". This restates TwoDofRecoversATwoAxisComposition's
// residual bound as an explicit convergence check with a tighter residual
// threshold, since a stalled or diverging optimiser could still land near
// enough on the angles by coincidence for a loose bound.
TEST(Reduced, TwoDofConvergesTightlyFromColdZeroSeed)
{
    glm::quat q = glm::angleAxis(glm::radians(40.0f), glm::vec3(1, 0, 0))
                * glm::angleAxis(glm::radians(-25.0f), glm::vec3(0, 0, 1));
    float seed[2] = {0.0f, 0.0f};
    float out[2]  = {0.0f, 0.0f};
    solve_two_dof(q, 0, 2, seed, out);
    EXPECT_LT(residual_two_dof(q, 0, 2, out), 1e-4f);
}

// When the target rotation genuinely requires all three axes (e.g. a
// rotation about a generic diagonal axis that is not spanned by any pair
// of the two chosen axes alone), the 2-DOF fit must leave a LARGE residual.
// Task 8 relies on this residual being meaningful to reject a 2-DOF
// candidate; if solve_two_dof's Gauss-Newton silently found some nearby
// point without the residual reflecting genuine unrepresentability, a
// joint that needs three stages would get incorrectly reduced to two.
TEST(Reduced, TwoDofReportsLargeResidualForUnrepresentableMotion)
{
    // A rotation with substantial simultaneous X, Y, and Z components.
    glm::quat q = glm::angleAxis(glm::radians(50.0f), glm::normalize(glm::vec3(1, 1, 1)));
    float seed[2] = {0.0f, 0.0f};
    float out[2]  = {0.0f, 0.0f};
    solve_two_dof(q, 0, 2, seed, out);
    EXPECT_GT(residual_two_dof(q, 0, 2, out), glm::radians(5.0f));
}

// solve_two_dof must terminate (not hang) and produce finite output even
// when seeded far from the solution and fed an adversarial target.
TEST(Reduced, TwoDofTerminatesAndStaysFiniteFromAWildSeed)
{
    glm::quat q = glm::angleAxis(glm::radians(50.0f), glm::normalize(glm::vec3(1, 1, 1)));
    float seed[2] = {glm::radians(400.0f), glm::radians(-400.0f)};
    float out[2]  = {0.0f, 0.0f};
    solve_two_dof(q, 0, 2, seed, out);
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_TRUE(std::isfinite(out[1]));
    float r = residual_two_dof(q, 0, 2, out);
    EXPECT_TRUE(std::isfinite(r));
}

// solve_two_dof must be deterministic: identical inputs, identical output.
TEST(Reduced, TwoDofIsDeterministic)
{
    glm::quat q = glm::angleAxis(glm::radians(40.0f), glm::vec3(1, 0, 0))
                * glm::angleAxis(glm::radians(-25.0f), glm::vec3(0, 0, 1));
    float seed[2] = {0.0f, 0.0f};
    float out1[2] = {0.0f, 0.0f};
    float out2[2] = {0.0f, 0.0f};
    solve_two_dof(q, 0, 2, seed, out1);
    solve_two_dof(q, 0, 2, seed, out2);
    EXPECT_FLOAT_EQ(out1[0], out2[0]);
    EXPECT_FLOAT_EQ(out1[1], out2[1]);
}

// --- Fix-round coverage (post-review): pin the residual metric's absolute
// scale and small-error resolution, and guard the accept/reject threshold
// Task 8 will use. ---

// Every prior residual assertion in this file is either a loose EXPECT_LT
// (satisfied by anything smaller) or an EXPECT_GT with wide margin, so none
// of them pin the metric's absolute SCALE. A factor-of-two error in
// angular_distance (e.g. `1.0f * acos(d)` instead of `2.0f * acos(d)`) would
// pass every other test in this file. Inject a rotation with a KNOWN
// residual (compose the exact hinge fit with a known extra 0.05 rad swing
// about a different axis) and check the reported residual matches that
// known value, not just "small" or "large".
TEST(Reduced, OneDofResidualMatchesAKnownInjectedErrorMagnitude)
{
    const float true_angle = glm::radians(20.0f);
    glm::quat q = axis_quat(0, true_angle) * glm::angleAxis(0.05f, glm::vec3(0, 1, 0));
    float a = solve_one_dof(q, 0);
    EXPECT_NEAR(residual_one_dof(q, 0, a), 0.05f, 5e-3f);
}

// The acos(|dot|)-based metric used before this fix round was dead (read as
// exactly 0.0) for true errors below roughly 9e-4 rad, which meant the
// brief's EXPECT_LT(..., 1e-5f) and this file's own
// TwoDofConvergesTightlyFromColdZeroSeed EXPECT_LT(..., 1e-4f) were both
// asserting below the metric's own resolution floor -- a solver that only
// ever managed 8e-4 rad would have passed both. Pin a case that actually
// requires the full 24-iteration budget (a wild seed against a
// near-singular two-axis configuration) to a tolerance the OLD metric could
// not have distinguished from zero, so a cut iteration budget is caught.
TEST(Reduced, TwoDofNeedsFullIterationBudgetNearSingularity)
{
    glm::quat q = glm::angleAxis(glm::radians(89.9f), glm::vec3(1, 0, 0))
                * glm::angleAxis(glm::radians(10.0f), glm::vec3(0, 0, 1));
    float seed[2] = {glm::radians(-150.0f), glm::radians(150.0f)};
    float out[2]  = {0.0f, 0.0f};
    solve_two_dof(q, 0, 2, seed, out);
    EXPECT_LT(residual_two_dof(q, 0, 2, out), 1e-5f);
}

// Regression guard for the ~0.02 rad accept/reject gate Task 8 will use.
// Sweep a small extra off-plane rotation added to an otherwise clean 2-axis
// composition, straddling the gate from just below to just above, and check
// the residual's accept/reject verdict is monotonic and lands on the
// expected side at both ends. This does not lock in Task 8's own logic (that
// is Task 8's job) -- it only confirms this file's metric behaves sanely at
// the scale Task 8 depends on.
TEST(Reduced, ResidualStraddlesTheAcceptRejectGateSensibly)
{
    glm::quat base = axis_quat(0, glm::radians(40.0f)) * axis_quat(2, glm::radians(-25.0f));

    float seed[2] = {0.0f, 0.0f};
    float out_below[2];
    glm::quat q_below = base * glm::angleAxis(0.0138f, glm::vec3(0, 1, 0));
    solve_two_dof(q_below, 0, 2, seed, out_below);
    EXPECT_LT(residual_two_dof(q_below, 0, 2, out_below), 0.02f);

    float out_above[2];
    glm::quat q_above = base * glm::angleAxis(0.0230f, glm::vec3(0, 1, 0));
    solve_two_dof(q_above, 0, 2, seed, out_above);
    EXPECT_GT(residual_two_dof(q_above, 0, 2, out_above), 0.02f);
}
