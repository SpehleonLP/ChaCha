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

// Contract: with a single frame there is no transition history to run the
// Viterbi tie-break over, so resolve_branches applies the same nearest-rest
// criterion directly to the two candidate branches. For this input,
// solve_euler's own principal branch (10, 20, 30) deg already IS the
// nearest-rest branch (its alternate is (190, 160, 210) deg, far from
// rest), so this also happens to equal what an unconditional "always
// branch 0" rule would have produced -- this test alone cannot distinguish
// the two policies; see SingleFrameResolvesNearestRestBranch below for a
// case where they differ.
TEST(DP, SingleFramePassesThroughNearestRestBranchUnchanged)
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

// Pins the n == 1 nearest-rest behaviour where it actually differs from
// "always take solve_euler's own branch 0": a single frame at (179, 89,
// 179) degrees. Its own principal branch has squared rest-distance
// (179^2 + 89^2 + 179^2) in degrees-squared ~= 72243 (deg^2) -- i.e. very
// far from rest -- while its alternate_branch, (-1, 91, -1) degrees, has
// squared rest-distance ~1+91^2+1 ~= 8283 deg^2, also far in absolute
// terms but decisively closer than the principal branch's. Converted to
// radians (as resolve_branches actually computes it) principal's squared
// distance is ~22.0 against the alternate's ~2.52 -- confirmed by direct
// computation below rather than asserted from the numbers alone. A resolver
// that always took branch 0 unconditionally (the pre-fix behaviour) would
// return the far-from-rest reading here; this test fails against that.
TEST(DP, SingleFrameResolvesNearestRestBranch)
{
    Chart c = tait_xyz();
    float angle[3] = {glm::radians(179.0f), glm::radians(89.0f), glm::radians(179.0f)};
    glm::quat q = compose_chart(c, angle);
    EulerSolution principal = solve_euler(q, c);
    EulerSolution alternate = alternate_branch(principal, c);

    auto sq_rest_dist = [](const EulerSolution& s) {
        float d = 0.0f;
        for (int a = 0; a < 3; ++a) {
            float w = principal_difference(s.angle[a], 0.0f);
            d += w * w;
        }
        return d;
    };
    const float principal_dist = sq_rest_dist(principal);
    const float alternate_dist = sq_rest_dist(alternate);
    // Pin the premise: the principal branch is NOT the nearest-rest one
    // here, so "always branch 0" and "nearest rest" genuinely disagree.
    ASSERT_GT(principal_dist, alternate_dist);

    std::vector<glm::quat> rel = {q};
    std::vector<float> times = {0.25f};
    Trajectory t = resolve_branches(rel, times, c);

    for (int a = 0; a < 3; ++a) {
        ASSERT_EQ(t.angle[a].size(), 1u);
        EXPECT_NEAR(t.angle[a][0], alternate.angle[a], 1e-5f) << "axis " << a;
    }
}

// resolve_branches has a second, independent ambiguity from the one
// anchor_trajectory fixes: alternate_branch is (a0+pi, pi-a1, a2+pi) for
// Tait-Bryan charts (and (a0+pi, -a1, a2+pi) for proper-Euler charts), and
// the Viterbi cost only ever sums *consecutive* differences, so those
// constant +-pi offsets cancel exactly on axes 0 and 2, and negating axis 1
// about pi leaves its square unchanged. That makes the whole-path cost of
// staying entirely on solve_euler's principal branch an exact tie (up to
// float rounding) with staying entirely on its alternate_branch twin, for
// every input on every chart -- not just pure single-axis rotations. Left
// unresolved, which family resolve_branches returns is decided by float
// rounding noise, and a joint at rest can be reported at (180,180,180)
// instead of (0,0,0) -- reconstructs fine, but is wrong Stage output before
// any cross-animation union even happens.
//
// The fix seeds the t==0 Viterbi cost with each branch's squared distance
// from the rest pose (see the seed loop in resolve_branches), so the
// near-rest family wins decisively. This test asserts that directly: a
// small sweep about a single axis must resolve with axes 1 and 2 near zero,
// not near +-180. It fails against the unpatched resolve_branches (verified:
// unpatched, this specific sweep landed on angle[1]==180, angle[2]==180).
// Verified directly which sweep to use here: the tie is decided by float
// rounding order, and it is NOT the same winner for every sweep. A -20 -> 20
// degree sweep happens to land on the near-rest family even unpatched (a
// false positive for this regression test, caught by running it against the
// unpatched code before trusting it); a 0 -> 30 degree sweep reliably lands
// on the far family (180, 180) unpatched, so that is the sweep used here.
TEST(DP, ResolvesToNearRestFamilyNotFarFamily)
{
    std::vector<glm::quat> rel;
    std::vector<float> times;
    for (int i = 0; i <= 20; ++i) {
        float f = static_cast<float>(i) / 20.0f;
        float deg = 0.0f + f * 30.0f;   // 0 -> 30 degrees about X
        times.push_back(i * 0.05f);
        rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
    }
    Trajectory t = resolve_branches(rel, times, tait_xyz());

    for (size_t i = 0; i < t.angle[1].size(); ++i) {
        EXPECT_LT(std::fabs(glm::degrees(t.angle[1][i])), 90.0f) << "at " << i;
        EXPECT_LT(std::fabs(glm::degrees(t.angle[2][i])), 90.0f) << "at " << i;
    }
}

// The seed this task replaced picked a family using only frame 0's distance
// from rest, which is wrong whenever frame 0 favours one family but the
// REST of the path favours the other: the family flip is a whole-path
// property (see the NOTE in resolve_branches), and a per-frame or seed-based
// heuristic cannot see that. This is the reviewer's counterexample: A sweeps
// (5,5,5) -> (60,40,60) degrees (near rest throughout, no ambiguity); B
// sweeps (170,85,170) -> (5,5,5) degrees, so frame 0 alone favours the FAR
// family (170,85,170 is closer to (180,90,180)'s neighbourhood than to
// rest... actually the reviewer's point is sharper: B's frame 0 principal
// reading is nearer to the ALTERNATE branch than to rest, even though B's
// path overall -- and certainly its last frame, (5,5,5) -- is near rest.
// A frame-0-only seed anchors B to the wrong family for its whole path
// before the DP ever runs; the whole-path post-pass instead looks at where
// B's reconstructed trajectory as a whole sits.
// Verified explicitly (see task-5-report.md fix-round-2 section): this
// test FAILS against the frame-0 seed (union ~235 degrees against the true
// 165), not merely against the fully-unpatched zero-seeded code.
TEST(DP, WholePathRestDistanceOutweighsFrameZero)
{
    Chart c = tait_xyz();
    auto sweep3 = [&](float x0, float y0, float z0, float x1, float y1, float z1) {
        std::vector<glm::quat> rel;
        std::vector<float> times;
        for (int i = 0; i <= 20; ++i) {
            float f = static_cast<float>(i) / 20.0f;
            float angle[3] = {
                glm::radians(x0 + f * (x1 - x0)),
                glm::radians(y0 + f * (y1 - y0)),
                glm::radians(z0 + f * (z1 - z0)),
            };
            rel.push_back(compose_chart(c, angle));
            times.push_back(i * 0.02f);
        }
        return std::make_pair(rel, times);
    };

    auto [ra, ta] = sweep3(5.0f, 5.0f, 5.0f, 60.0f, 40.0f, 60.0f);
    auto [rb, tb] = sweep3(170.0f, 85.0f, 170.0f, 5.0f, 5.0f, 5.0f);

    Trajectory a = resolve_branches(ra, ta, c);
    Trajectory b = resolve_branches(rb, tb, c);

    const float zero_ref[3] = {0.0f, 0.0f, 0.0f};
    anchor_trajectory(a, zero_ref);

    float a_lo = *std::min_element(a.angle[0].begin(), a.angle[0].end());
    float a_hi = *std::max_element(a.angle[0].begin(), a.angle[0].end());
    const float a_ref[3] = {0.5f * (a_lo + a_hi), 0.0f, 0.0f};
    anchor_trajectory(b, a_ref);

    float lo = std::min(*std::min_element(a.angle[0].begin(), a.angle[0].end()),
                        *std::min_element(b.angle[0].begin(), b.angle[0].end()));
    float hi = std::max(*std::max_element(a.angle[0].begin(), a.angle[0].end()),
                        *std::max_element(b.angle[0].begin(), b.angle[0].end()));

    // True excursion is 5..170 degrees, i.e. 165 degrees wide.
    EXPECT_NEAR(glm::degrees(hi - lo), 165.0f, 1.0f);
}

// The scenario anchor_trajectory exists for, exercised end to end: two
// independent animations of one joint, each pushed through resolve_branches
// (not constructed by hand), then anchored against a running reference
// before their ranges are unioned. This is the only executable coverage of
// the failure mode Tasks 8, 10 and 11 depend on -- without it the suite can
// stay green while that integration path is broken (as it was: before the
// resolve_branches seed fix, this exact test asserted a union under 40
// degrees and got 190; after the fix it gets 20, matching the true
// 170..190 degree excursion).
TEST(DP, AnchoringKeepsIndependentAnimationsCommensurable_ThroughResolveBranches)
{
    auto build = [](float from_deg, float to_deg) {
        std::vector<glm::quat> rel; std::vector<float> times;
        for (int i = 0; i <= 20; ++i) {
            float f = static_cast<float>(i) / 20.0f;
            float deg = from_deg + f * (to_deg - from_deg);
            times.push_back(i * 0.05f);
            rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
        }
        return std::make_pair(rel, times);
    };

    auto [ra, ta] = build(170.0f, 185.0f);   // crosses 180 upward
    auto [rb, tb] = build(190.0f, 175.0f);   // crosses 180 downward

    Trajectory a = resolve_branches(ra, ta, tait_xyz());
    Trajectory b = resolve_branches(rb, tb, tait_xyz());

    const float zero_ref[3] = {0.0f, 0.0f, 0.0f};
    anchor_trajectory(a, zero_ref);

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

// A more direct cross-animation range check that genuinely exercises the
// +-180 wrap boundary anchor_trajectory exists for (an earlier version of
// this test used 0->30 / -20->20 degree sweeps, neither of which crosses
// +-180 at all: both resolve_branches's family fix AND a no-op
// anchor_trajectory would pass it unioned as-is, since -20..30 needs no 2pi
// shift to read correctly. That made half the test's own comment -- "guards
// a wrap-boundary mismatch" -- vacuous. Verified directly: 170->195 /
// 200->220 degrees is not vacuous. 170->195 stays on the near side of the
// wrap (resolves to 170..195 directly); 200->220 reads on the FAR side
// (resolves to -160..-140, since solve_euler's principal branch wraps
// there), so the raw, un-anchored union is a spurious 355 degrees, and only
// with anchor_trajectory does it collapse to the true 170..220 degree
// (50 degree) excursion -- verified both numbers directly against this
// exact construction.
TEST(DP, CrossAnimationRangeUnionMatchesTrueExcursion)
{
    auto build = [](float from_deg, float to_deg) {
        std::vector<glm::quat> rel; std::vector<float> times;
        for (int i = 0; i <= 20; ++i) {
            float f = static_cast<float>(i) / 20.0f;
            float deg = from_deg + f * (to_deg - from_deg);
            times.push_back(i * 0.05f);
            rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
        }
        return std::make_pair(rel, times);
    };

    auto [rc, tc] = build(170.0f, 195.0f);
    auto [rd, td] = build(200.0f, 220.0f);

    Trajectory c = resolve_branches(rc, tc, tait_xyz());
    Trajectory d = resolve_branches(rd, td, tait_xyz());

    // Without anchoring, the raw union is spurious: c reads 170..195, d
    // reads on the other side of the wrap (-160..-140), so their raw union
    // spans ~355 degrees. This is not itself an assertion (anchor_trajectory
    // is the function under test, not resolve_branches's raw output), but
    // pins the premise that this construction actually needs the anchor.
    float raw_lo = std::min(*std::min_element(c.angle[0].begin(), c.angle[0].end()),
                            *std::min_element(d.angle[0].begin(), d.angle[0].end()));
    float raw_hi = std::max(*std::max_element(c.angle[0].begin(), c.angle[0].end()),
                            *std::max_element(d.angle[0].begin(), d.angle[0].end()));
    ASSERT_GT(glm::degrees(raw_hi - raw_lo), 300.0f);

    const float zero_ref[3] = {0.0f, 0.0f, 0.0f};
    anchor_trajectory(c, zero_ref);

    float c_lo = *std::min_element(c.angle[0].begin(), c.angle[0].end());
    float c_hi = *std::max_element(c.angle[0].begin(), c.angle[0].end());
    const float c_ref[3] = {0.5f * (c_lo + c_hi), 0.0f, 0.0f};
    anchor_trajectory(d, c_ref);

    float lo = std::min(*std::min_element(c.angle[0].begin(), c.angle[0].end()),
                        *std::min_element(d.angle[0].begin(), d.angle[0].end()));
    float hi = std::max(*std::max_element(c.angle[0].begin(), c.angle[0].end()),
                        *std::max_element(d.angle[0].begin(), d.angle[0].end()));

    // True excursion is 170..220 degrees, i.e. 50 degrees wide.
    EXPECT_NEAR(glm::degrees(hi - lo), 50.0f, 1.0f);
}

// Unit-level coverage of anchor_trajectory itself, independent of
// resolve_branches: two synthetic Trajectory objects built directly with
// the exact raw ranges the cross-animation scenario above depends on
// (one on each side of the +-180 wrap), so this test's pass/fail is
// determined solely by anchor_trajectory's own 2pi-shift logic.
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

// anchor_trajectory loops over all three axes independently; the tests
// above only ever populate axis 0. This covers more than one axis carrying
// a non-trivial trajectory in the same call, with different reference
// values and different required shifts per axis, so a bug that only
// touched axis 0 (or that shared state across axes) would be caught.
TEST(DP, AnchorsMultipleAxesIndependently)
{
    Trajectory t;
    t.time = {0.0f, 1.0f};
    // Axis 0: same as the direct-construction commensurability test above --
    // raw range -185..-170 degrees, needs +360 to align near +177.5.
    t.angle[0] = {glm::radians(-170.0f), glm::radians(-185.0f)};
    // Axis 1: already well inside (-pi, pi], needs no shift at all.
    t.angle[1] = {glm::radians(10.0f), glm::radians(20.0f)};
    // Axis 2: sits just past pi, needs a single -2pi shift with a non-zero
    // reference, distinct from axis 0's +2pi shift, to prove the two axes
    // are not accidentally sharing one shift value.
    t.angle[2] = {glm::radians(185.0f), glm::radians(195.0f)};

    const float reference[3] = {glm::radians(177.5f), glm::radians(0.0f), glm::radians(-175.0f)};
    anchor_trajectory(t, reference);

    // Axis 0 shifted by +360 degrees.
    EXPECT_NEAR(glm::degrees(t.angle[0][0]), 190.0f, 1e-2f);
    EXPECT_NEAR(glm::degrees(t.angle[0][1]), 175.0f, 1e-2f);

    // Axis 1 unshifted.
    EXPECT_NEAR(glm::degrees(t.angle[1][0]), 10.0f, 1e-2f);
    EXPECT_NEAR(glm::degrees(t.angle[1][1]), 20.0f, 1e-2f);

    // Axis 2 shifted by -360 degrees.
    EXPECT_NEAR(glm::degrees(t.angle[2][0]), -175.0f, 1e-2f);
    EXPECT_NEAR(glm::degrees(t.angle[2][1]), -165.0f, 1e-2f);
}
