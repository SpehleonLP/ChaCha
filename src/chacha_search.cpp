#include "chacha_internal.h"
#include <algorithm>
#include <cmath>

namespace ChaCha {
namespace detail {

namespace {

constexpr float kTwoPi = 2.0f * 3.14159265358979323846f;

StageType rotate_stage_for(int axis)
{
    switch (axis) {
    case 0:  return StageType::xRotate;
    case 1:  return StageType::yRotate;
    default: return StageType::zRotate;
    }
}

// Cross-animation anchoring protocol (Task 8 ruling 2): anchoring each
// animation of a joint independently against its own midpoint is exactly
// the defect Task 5's anchor_trajectory was introduced to fix. Two
// animations that straddle the 2pi wrap boundary from opposite sides must
// be anchored against a COMMON reference, or their ranges union to a
// spurious ~2pi span. The first animation of a candidate has no running
// range yet, so it anchors against an all-zero reference (reproducing the
// (-pi, pi] normalisation of a lone trajectory's own midpoint); every
// subsequent animation anchors against the per-axis midpoint of the
// running union accumulated so far, so it lands in the same 2pi branch as
// what came before it.
void reference_from_summary(const CandidateSummary& summary, float out[3])
{
    for (int a = 0; a < 3; ++a)
        out[a] = summary.axis_valid[a]
            ? 0.5f * (summary.min_value[a] + summary.max_value[a])
            : 0.0f;
}

// The cross-animation protocol above anchors each animation against the
// PRIOR running union, so the final reported range is only unique up to a
// 2pi shift per axis: it depends on which animation happened to be
// processed first (whichever one is first anchors against an all-zero
// reference and fixes the branch every later animation is pulled toward).
// Swapping the caller's animation order can swap which branch "wins",
// reporting the identical physical range 360 degrees apart -- caller-
// visible non-determinism in AGI_articulations min_value/max_value, which
// is not acceptable even though it's an inherent consequence of the
// protocol rather than a bug in it. Fix once, after the union is
// complete: shift each axis's final [min, max] by whichever multiple of
// 2pi brings its midpoint nearest zero. This preserves the range's width
// (so every width-based test and the scoring in total_range() are
// unaffected) while making the reported absolute values depend only on
// the physical motion, not on animation order. See
// Search.FinalRangeIsIndependentOfAnimationOrder.
//
// Deliberately NOT std::round for the nearest-multiple choice: std::round
// breaks an exact .5 tie away from zero, which is anti-symmetric in the
// input's sign (round(0.5) == 1, round(-0.5) == -1). A range whose true
// midpoint sits exactly at +-pi is precisely the case that lands on this
// tie, and the two animation orderings arrive at that midpoint with
// OPPOSITE signs (+pi from one order's raw union, -pi from the other's,
// per the protocol above) -- so std::round here would still leave the two
// orderings 2pi apart on exactly the input this function exists to fix.
// floor(x + 0.5) ("round half up") breaks the same tie the same way
// regardless of the input's sign, which is what makes the two orderings
// converge. Confirmed by Search.FinalRangeIsIndependentOfAnimationOrder,
// which is built specifically to land on this tie (a 10 degree arc
// centred exactly on the +-180 antipode).
float round_half_up(float x)
{
    return std::floor(x + 0.5f);
}

void normalize_range_to_nearest_zero(CandidateSummary& s)
{
    for (int a = 0; a < 3; ++a) {
        if (!s.axis_valid[a]) continue;
        const float mid   = 0.5f * (s.min_value[a] + s.max_value[a]);
        const float shift = kTwoPi * round_half_up((0.0f - mid) / kTwoPi);
        if (shift == 0.0f) continue;
        s.min_value[a] += shift;
        s.max_value[a] += shift;
    }
}

// Full-rank candidate: exact solve, Viterbi, anchor, summarise.
Candidate evaluate_chart(const Chart& chart, const JointMotion& m, const Options& o)
{
    Candidate c;
    c.dof    = 3;
    c.proper = chart.proper;
    for (int s = 0; s < 3; ++s) { c.stage[s] = chart.stage[s]; c.axis[s] = chart.axis[s]; }

    for (size_t ai = 0; ai < m.rel_by_animation.size(); ++ai) {
        Trajectory t = resolve_branches(m.rel_by_animation[ai], m.times_by_animation[ai], chart);
        if (t.time.empty()) continue;

        // NOTE on test coverage (Task 8 review finding 5): this call site
        // applies the identical cross-animation anchoring protocol as
        // evaluate_one_dof/evaluate_two_dof, but there is no committed test
        // that can actually observe a regression here. Mutating this
        // reference to all-zero (the same "anchor every animation
        // independently" defect the wrap tests above pin) produced ZERO
        // output difference across 200 randomised two-animation joint
        // motions plus a directed wrap construction: whichever chart's
        // range blows up to a spurious ~2pi span from bad anchoring simply
        // loses select_candidate's min-total-range tie-break to a
        // different, unaffected chart, and the broken chart's own output
        // is silently discarded rather than surfaced as the winner. The
        // shared protocol IS covered -- by
        // Search.TwoDofCrossAnimationAnchorCollapsesOppositeSidesOfTheWrap
        // in chacha_search_test.cpp, since evaluate_two_dof has no such
        // masking tie-break (dof strictly dominates scoring, so a 2-DOF
        // winner's own reported range is never substituted out for a
        // different pair's). Do not treat the absence of a
        // chart-path-specific test as evidence this code path is untested;
        // it is exercised by every chart-producing test above, just not in
        // a way that could catch THIS SPECIFIC regression if reintroduced
        // here.
        float reference[3];
        reference_from_summary(c.summary, reference);
        anchor_trajectory(t, reference);

        // Any quaternion decomposes exactly into 3 angles under any chart
        // (three distinct axes always exactly span SO(3), including the 6
        // proper-Euler charts with a repeated axis -- see
        // Charts.SolveRoundTripsExactlyForEveryChart and
        // Charts.SolveRoundTripsAtSingularConfigurations), so this is not
        // assumed to be exactly 0: it is the actual measured round-trip
        // error, via the same well-conditioned metric residual_one_dof/
        // residual_two_dof use, expected to land in the ~1e-7 to ~1e-4 rad
        // band those two chart tests already establish as this solve's
        // float round-off floor.
        float worst = 0.0f;
        for (size_t i = 0; i < t.angle[0].size(); ++i) {
            const float angle[3] = {t.angle[0][i], t.angle[1][i], t.angle[2][i]};
            const glm::quat recon = compose_chart(chart, angle);
            worst = std::max(worst, quaternion_angular_distance(recon, m.rel_by_animation[ai][i]));
        }
        union_into(c.summary, summarise(t, worst, o));
    }
    normalize_range_to_nearest_zero(c.summary);
    return c;
}

Candidate evaluate_one_dof(int axis, const JointMotion& m, const Options& o)
{
    Candidate c;
    c.dof      = 1;
    c.stage[0] = rotate_stage_for(axis);
    c.axis[0]  = axis;

    for (size_t ai = 0; ai < m.rel_by_animation.size(); ++ai) {
        const auto& rel   = m.rel_by_animation[ai];
        const auto& times = m.times_by_animation[ai];
        // Truncate to whichever of rel/times is shorter, matching
        // resolve_branches's own defined policy for mismatched-length
        // per-animation arrays (chacha_dp.cpp): process only as many
        // frames as both spans actually provide, rather than reading rel
        // past the end of times (or vice versa).
        const size_t n = std::min(rel.size(), times.size());
        if (n == 0) continue;

        Trajectory t;
        t.time.assign(times.begin(), times.begin() + n);
        // Trajectory::worst_conditioning defaults to 1.0f ("perfectly
        // conditioned"), a real value chart_conditioning() computes for
        // 3-DOF chart candidates. solve_one_dof has no analogous
        // conditioning measure at all, so leaving the default in place
        // would silently fabricate a perfect score for every one-DOF
        // candidate rather than reporting "not measured". Use an explicit
        // sentinel outside chart_conditioning()'s real [0, 1] range instead
        // -- see the same choice in evaluate_two_dof.
        t.worst_conditioning = -1.0f;
        // Slot 0 holds this candidate's one DOF (position/stage order, the
        // same convention evaluate_chart uses via resolve_branches's
        // per-chart-stage Trajectory). Slots 1 and 2 are left empty so
        // summarise/union_into correctly mark them as having no observed
        // samples (axis_valid == false) instead of fabricating a spurious
        // [0, 0] range for DOFs this candidate doesn't use.
        t.angle[0].assign(n, 0.0f);

        float worst = 0.0f;
        float prev  = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float raw = solve_one_dof(rel[i], axis);
            const float ang = (i == 0) ? raw : prev + principal_difference(raw, prev);
            t.angle[0][i] = ang;
            prev = ang;
            worst = std::max(worst, residual_one_dof(rel[i], axis, ang));
        }

        float reference[3];
        reference_from_summary(c.summary, reference);
        anchor_trajectory(t, reference);
        union_into(c.summary, summarise(t, worst, o));
    }
    normalize_range_to_nearest_zero(c.summary);
    return c;
}

Candidate evaluate_two_dof(int axis0, int axis1, const JointMotion& m, const Options& o)
{
    Candidate c;
    c.dof      = 2;
    c.stage[0] = rotate_stage_for(axis0);  c.axis[0] = axis0;
    c.stage[1] = rotate_stage_for(axis1);  c.axis[1] = axis1;

    for (size_t ai = 0; ai < m.rel_by_animation.size(); ++ai) {
        const auto& rel   = m.rel_by_animation[ai];
        const auto& times = m.times_by_animation[ai];
        // See the matching comment in evaluate_one_dof: truncate to
        // whichever of rel/times is shorter, matching resolve_branches's
        // policy for mismatched-length per-animation arrays.
        const size_t n = std::min(rel.size(), times.size());
        if (n == 0) continue;

        Trajectory t;
        t.time.assign(times.begin(), times.begin() + n);
        // See evaluate_one_dof: solve_two_dof has no conditioning measure
        // either, so use the same explicit "not measured" sentinel rather
        // than leaving Trajectory's default 1.0f ("perfectly conditioned")
        // in place, which would fabricate a perfect score.
        t.worst_conditioning = -1.0f;
        // Slots 0 and 1 hold this candidate's two DOFs in position order
        // (slot 0 = axis0, slot 1 = axis1); slot 2 is left empty (see the
        // comment in evaluate_one_dof above).
        for (int a = 0; a < 2; ++a) t.angle[a].assign(n, 0.0f);

        float worst   = 0.0f;
        float seed[2] = {0.0f, 0.0f};
        float prev[2] = {0.0f, 0.0f};

        for (size_t i = 0; i < n; ++i) {
            float sol[2];
            // Seeding from the previous frame keeps the solve in the same
            // branch (solve_two_dof never returns a point worse than its
            // seed).
            solve_two_dof(rel[i], axis0, axis1, seed, sol);
            seed[0] = sol[0];
            seed[1] = sol[1];

            for (int a = 0; a < 2; ++a) {
                const float ang = (i == 0) ? sol[a] : prev[a] + principal_difference(sol[a], prev[a]);
                t.angle[a][i] = ang;
                prev[a] = ang;
            }
            worst = std::max(worst, residual_two_dof(rel[i], axis0, axis1, sol));
        }

        float reference[3];
        reference_from_summary(c.summary, reference);
        anchor_trajectory(t, reference);
        union_into(c.summary, summarise(t, worst, o));
    }
    normalize_range_to_nearest_zero(c.summary);
    return c;
}

// CandidateSummary's slots follow each candidate's own position/stage
// order (slot i is this candidate's i-th DOF, stage[i]/axis[i]), not the
// physical rotation axis — consistent across evaluate_chart (which
// inherits slot order from resolve_branches's per-chart-stage Trajectory)
// and evaluate_one_dof/evaluate_two_dof above. So only slots [0, dof) are
// ever meaningful; total_range must not read past dof.
float total_range(const Candidate& c)
{
    float sum = 0.0f;
    for (int a = 0; a < c.dof; ++a)
        sum += c.summary.max_value[a] - c.summary.min_value[a];
    return sum;
}

// Lexicographic: fewer DOF, then tighter total range, then better conditioning.
bool better(const Candidate& lhs, const Candidate& rhs)
{
    if (!rhs.summary.valid) return true;
    if (!lhs.summary.valid) return false;
    if (lhs.dof != rhs.dof) return lhs.dof < rhs.dof;

    const float lr = total_range(lhs), rr = total_range(rhs);
    if (std::fabs(lr - rr) > 1e-4f) return lr < rr;

    return lhs.summary.worst_conditioning > rhs.summary.worst_conditioning;
}

} // namespace

Candidate select_candidate(const JointMotion& motion, const Options& options)
{
    Candidate best;

    // Reduced candidates are only admissible if they explain every observed
    // pose within the fit-residual gate.
    for (int axis = 0; axis < 3; ++axis) {
        Candidate c = evaluate_one_dof(axis, motion, options);
        if (c.summary.valid && c.summary.max_residual_rad <= options.max_fit_residual_rad
            && better(c, best))
            best = c;
    }

    for (int a0 = 0; a0 < 3; ++a0)
        for (int a1 = 0; a1 < 3; ++a1) {
            if (a0 == a1) continue;
            Candidate c = evaluate_two_dof(a0, a1, motion, options);
            if (c.summary.valid && c.summary.max_residual_rad <= options.max_fit_residual_rad
                && better(c, best))
                best = c;
        }

    for (const auto& chart : all_charts()) {
        Candidate c = evaluate_chart(chart, motion, options);
        if (c.summary.valid && better(c, best)) best = c;
    }

    return best;
}

} // namespace detail
} // namespace ChaCha
