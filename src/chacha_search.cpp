#include "chacha_internal.h"
#include <algorithm>
#include <cmath>

namespace ChaCha {
namespace detail {

namespace {

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

        float reference[3];
        reference_from_summary(c.summary, reference);
        anchor_trajectory(t, reference);

        // Three distinct axes reconstruct exactly (Charts.SolveRoundTrips-
        // ExactlyForEveryChart and Charts.SolveRoundTripsAtSingularConfig-
        // urations verified this for all 12 charts, including the 6
        // proper-Euler ones with a repeated axis), so the residual here is
        // 0 by construction rather than an approximation: any quaternion
        // decomposes exactly into 3 angles under any of these charts.
        union_into(c.summary, summarise(t, 0.0f, o));
    }
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
        if (rel.empty()) continue;

        Trajectory t;
        t.time.assign(times.begin(), times.end());
        // Slot 0 holds this candidate's one DOF (position/stage order, the
        // same convention evaluate_chart uses via resolve_branches's
        // per-chart-stage Trajectory). Slots 1 and 2 are left empty so
        // summarise/union_into correctly mark them as having no observed
        // samples (axis_valid == false) instead of fabricating a spurious
        // [0, 0] range for DOFs this candidate doesn't use.
        t.angle[0].assign(rel.size(), 0.0f);

        float worst = 0.0f;
        float prev  = 0.0f;
        for (size_t i = 0; i < rel.size(); ++i) {
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
        if (rel.empty()) continue;

        Trajectory t;
        t.time.assign(times.begin(), times.end());
        // Slots 0 and 1 hold this candidate's two DOFs in position order
        // (slot 0 = axis0, slot 1 = axis1); slot 2 is left empty (see the
        // comment in evaluate_one_dof above).
        for (int a = 0; a < 2; ++a) t.angle[a].assign(rel.size(), 0.0f);

        float worst   = 0.0f;
        float seed[2] = {0.0f, 0.0f};
        float prev[2] = {0.0f, 0.0f};

        for (size_t i = 0; i < rel.size(); ++i) {
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
