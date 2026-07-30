#include "chacha_internal.h"
#include <algorithm>
#include <cmath>
#include <limits>

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

float total_rotation_angle(const glm::quat& q)
{
    return quaternion_angular_distance(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), q);
}

// One joint axis's accumulated phase: which axis, and a CandidateSummary
// whose slot 0 (and only slot 0) carries the observed range/velocity/
// acceleration for that axis. Slot 0 is used regardless of which final
// stage-order position this phase ends up in; solve_configuration copies
// slot 0 into the right final slot once the order is known.
struct AxisPhase {
    int              axis{-1};
    CandidateSummary summary;
};

// One in-progress run: a maximal span of consecutive frames whose
// dominant axis is `axis`, not yet closed by a return to rest.
//
// `prev_angle`/`have_prev` carry the running unwrap state (see the
// comment on solve_one_dof's (-pi, pi] range in chacha_internal.h): a
// sweep past +-180 degrees would otherwise report a wildly wrong range,
// since solve_one_dof re-wraps every frame independently. Unwrapping is
// reset per run (not carried across separate runs, even of the same
// axis): each run is anchored at its own start, so unwrapping fresh from
// that anchor is exactly as correct as unwrapping continuously across the
// whole animation would be, and simpler to reason about per-run.
struct OpenRun {
    bool                active{false};
    int                 axis{-1};
    std::vector<float>  times;
    std::vector<float>  angles;
    float               worst_residual{0.0f};
    bool                have_prev{false};
    float               prev_angle{0.0f};
};

// Closes `run` (if any) into `order`, merging into an existing same-axis
// entry (see accumulate_animation's own re-entry rule for why a same-axis
// re-entry within one animation is a merge, not a rejection).
bool close_run(OpenRun& run, std::vector<AxisPhase>& order, const Options& options)
{
    if (!run.active) return true;

    int found = -1;
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i].axis == run.axis) { found = static_cast<int>(i); break; }
    }
    if (found >= 0 && found != static_cast<int>(order.size()) - 1) {
        return false;   // a different axis intervened since this axis last ran
    }

    Trajectory t;
    t.time               = run.times;
    t.angle[0]            = run.angles;
    // No chart underlies a configuration-animation phase (it's a direct
    // single-axis read, not an Euler/swing-twist decomposition), so there
    // is no conditioning measure to report. Use the same explicit "not
    // measured" sentinel evaluate_one_dof/evaluate_two_dof use for the
    // same reason (see chacha_search.cpp), instead of Trajectory's default
    // 1.0f ("perfectly conditioned"), which would fabricate a perfect
    // score no chart search actually computed.
    t.worst_conditioning = -1.0f;

    const CandidateSummary s = summarise(t, run.worst_residual, options);
    if (found == -1) {
        order.push_back(AxisPhase{run.axis, s});
    } else {
        union_into(order[found].summary, s);
    }

    run = OpenRun{};
    return true;
}

// Walks ONE animation's rest-relative rotation samples, splitting into
// per-axis runs at every return to rest and merging same-axis runs back
// together (a phase that passes back through rest at its own midpoint,
// e.g. a sinusoidal sweep, is observed as two same-axis sub-runs and must
// merge, not be rejected as if the axis recurred after a different axis
// intervened -- see the module-level ruling this file follows,
// transcribed onto is_configuration_motion's declaration in
// chacha_internal.h).
//
// Returns false the moment either (a) a frame's rotation isn't well
// explained by any single axis (two axes genuinely simultaneous), or (b)
// an axis's run is interrupted mid-phase by a different axis becoming
// dominant without a return to rest, or (c) close_run finds genuine
// interleaving WITHIN this one animation (an axis recurring after a
// different axis intervened, e.g. X, Z, X).
//
// Deliberately does NOT carry `order`/interleaving state across separate
// animations: extract_phases below combines multiple animations' results
// by axis identity and a "first valid animation sets the order" rule
// instead, not by concatenating raw frames through one shared run/order
// state. See extract_phases for why.
bool accumulate_animation(
    const std::vector<glm::quat>& rel,
    const std::vector<float>&     times,
    const Options&                options,
    std::vector<AxisPhase>&       order)
{
    const size_t n = std::min(rel.size(), times.size());
    OpenRun run;
    // Time of the most recent rest sample seen (valid whenever `have_rest`
    // is true), so a run that opens straight from rest can be anchored at
    // exactly 0 rather than starting from its first active sample. Without
    // this, a phase whose data never happens to include a frame back at
    // rest (e.g. only two keyframes: rest, then the sweep's peak, with no
    // return) reports a spuriously collapsed range -- min == max at the
    // single observed active sample -- instead of the true rest-to-peak
    // span, and a range of exactly 0 gets discarded entirely by the noise
    // filter downstream.
    bool  have_rest  = false;
    float rest_time  = 0.0f;

    // options.rotation_threshold_rad also serves as the at-rest epsilon
    // for phase segmentation here. It is documented elsewhere (CLAUDE.md
    // step 6) as the noise floor for discarding negligible-motion stages
    // downstream in filter_stages -- a different purpose from "how close
    // to identity counts as returning to rest." Reusing it rather than
    // introducing a second, uncoordinated magic number is deliberate, but
    // it does couple the two: a caller raising this threshold to silence
    // noisy stages elsewhere will also make phase detection coarser here,
    // potentially merging or missing phases whose peak sits close to the
    // (now larger) rest band. No test in this file exercises that
    // coupling directly; it is a documented trade-off, not a resolved one.
    for (size_t i = 0; i < n; ++i) {
        const glm::quat q     = glm::normalize(rel[i]);
        const float     total = total_rotation_angle(q);

        if (total < options.rotation_threshold_rad) {
            if (run.active) {
                // Anchor the end of the sweep at exactly 0 too. Unwrapped
                // through the same principal_difference step as every
                // other sample (rather than a hard-coded 0.0f) so a
                // large-angle sweep's endpoint stays consistent with the
                // continuously unwrapped trajectory that led up to it,
                // instead of snapping back into (-pi, pi] at the last
                // instant.
                const float raw = 0.0f;   // the rest sample itself is ~0 on any axis
                const float ang = run.have_prev
                                ? run.prev_angle + principal_difference(raw, run.prev_angle)
                                : raw;
                run.times.push_back(times[i]);
                run.angles.push_back(ang);
            }
            if (!close_run(run, order, options)) return false;
            have_rest = true;
            rest_time = times[i];
            continue;
        }

        int   best_axis     = 0;
        float best_angle    = 0.0f;
        float best_residual = std::numeric_limits<float>::infinity();
        for (int a = 0; a < 3; ++a) {
            const float angle    = solve_one_dof(q, a);
            const float residual = residual_one_dof(q, a, angle);
            if (residual < best_residual) {
                best_residual = residual;
                best_axis     = a;
                best_angle    = angle;
            }
        }

        // Genuinely simultaneous multi-axis rotation (e.g. ordinary,
        // non-configuration motion) cannot be explained by any single
        // axis to within the same fit-quality gate the reduced-DOF search
        // uses elsewhere.
        if (best_residual > options.max_fit_residual_rad) return false;

        if (run.active && run.axis != best_axis) {
            // The dominant axis changed mid-phase with no return to rest
            // in between: two different axes active within one phase.
            return false;
        }
        if (!run.active) {
            run.active = true;
            run.axis   = best_axis;
            if (have_rest) {
                // Anchor the start of the sweep at exactly 0 too.
                run.times.push_back(rest_time);
                run.angles.push_back(0.0f);
                run.have_prev  = true;
                run.prev_angle = 0.0f;
            }
        }

        // solve_one_dof re-wraps every frame independently into
        // (-pi, pi], so a sweep that passes 180 degrees would otherwise
        // read as snapping to the opposite sign instead of continuing to
        // grow. Unwrap by accumulating the principal (shortest-path)
        // difference from the previous frame, exactly as
        // evaluate_one_dof does in chacha_search.cpp for the same reason.
        const float ang = run.have_prev
                         ? run.prev_angle + principal_difference(best_angle, run.prev_angle)
                         : best_angle;
        run.have_prev  = true;
        run.prev_angle = ang;

        run.times.push_back(times[i]);
        run.angles.push_back(ang);
        run.worst_residual = std::max(run.worst_residual, best_residual);
    }

    return close_run(run, order, options);
}

// Extracts the ordered, merged per-axis phases across every animation in
// `motion`.
//
// Stage order is this task's entire deliverable, and for a SINGLE
// configuration animation it is recovered directly from first-encounter
// order within that one clip (accumulate_animation above). For MULTIPLE
// animations (e.g. a joint driven by two separately-authored AGI clips,
// such as the "AGI Configuration" / "AGI Configuration.001" pair some
// real models split a joint's phases across), concatenating each
// animation's frames through one shared run/order state -- the earlier
// version of this function -- made the reported order an artifact of
// `rel_by_animation`'s arbitrary index order: swapping which clip came
// first in that vector silently flipped the reported stage order for
// otherwise-identical input, with no basis in anything the artist
// actually authored.
//
// Instead: each animation is validated independently (its own frames must
// still form clean single-axis phases, or the whole joint is rejected --
// same rule as before, just not spanning animation boundaries). The FIRST
// animation that is both valid and non-empty (i.e. the joint actually
// moves in it) supplies the authoritative stage order. Every other valid
// animation's phases are folded in by axis identity: a shared axis has its
// range/velocity/acceleration unioned into the first's entry (via the same
// union_into used everywhere else in this pipeline), and an axis the first
// animation never exercised is appended after the authoritative order
// (new information, not a contradiction of it). If two animations
// disagree about a shared axis's ORDER, the first animation's order wins
// outright and this is not treated as an error -- only a genuine
// single-axis-fit violation or intra-animation interleaving rejects the
// whole joint.
bool extract_phases(const JointMotion& motion, const Options& options,
                     std::vector<AxisPhase>& order)
{
    order.clear();
    bool have_authoritative = false;

    for (size_t ai = 0; ai < motion.rel_by_animation.size(); ++ai) {
        std::vector<AxisPhase> local;
        if (!accumulate_animation(motion.rel_by_animation[ai],
                                   motion.times_by_animation[ai],
                                   options, local)) {
            return false;
        }
        if (local.empty()) continue;   // this clip is locked; contributes nothing

        if (!have_authoritative) {
            order              = std::move(local);
            have_authoritative = true;
            continue;
        }

        for (const AxisPhase& p : local) {
            auto it = std::find_if(order.begin(), order.end(),
                                    [&](const AxisPhase& a) { return a.axis == p.axis; });
            if (it == order.end()) order.push_back(p);          // new axis: append
            else                   union_into(it->summary, p.summary);  // shared axis: union range
        }
    }

    return true;
}

} // namespace

bool is_configuration_motion(const JointMotion& motion, const Options& options)
{
    if (motion.rel_by_animation.empty()) return false;

    std::vector<AxisPhase> order;
    if (!extract_phases(motion, options, order)) return false;
    return !order.empty();
}

Candidate solve_configuration(const JointMotion& motion, const Options& options)
{
    Candidate c;

    std::vector<AxisPhase> order;
    if (!extract_phases(motion, options, order) || order.empty()) {
        // Locked joint: no phases observed anywhere. Set the same "not
        // measured" sentinel the populated branch below uses, rather than
        // leaving CandidateSummary's default 1.0f ("perfectly
        // conditioned") in place on an otherwise-invalid (summary.valid ==
        // false) result. Callers already gate on `valid` before looking at
        // this field, so this is unreachable in practice; setting it
        // anyway keeps the invariant "this path never reports 1.0f"
        // exception-free rather than "true except when dof == 0".
        c.summary.worst_conditioning = -1.0f;
        return c;
    }

    // order.size() cannot exceed 3: extract_phases/accumulate_animation
    // only ever create one AxisPhase per distinct axis value (0, 1, 2).
    c.dof = static_cast<int>(order.size());
    for (int s = 0; s < c.dof; ++s) {
        const AxisPhase& ph = order[s];
        c.stage[s] = rotate_stage_for(ph.axis);
        c.axis[s]  = ph.axis;

        c.summary.valid               = true;
        c.summary.axis_valid[s]       = true;
        c.summary.min_value[s]        = ph.summary.min_value[0];
        c.summary.max_value[s]        = ph.summary.max_value[0];
        // Velocity/acceleration are computed for real (via summarise's
        // resample-and-differentiate pipeline, same as every other
        // candidate path), not zeroed or sentinel'd -- consistent with
        // CLAUDE.md's "ChaCha reports observed ROM, not possible ROM":
        // every value this library reports already means "what this
        // animation's keyframes showed," never a physical limit, and a
        // configuration animation's sweep speed is no different in kind.
        //
        // It IS different in DEGREE, and that must be documented rather
        // than left implicit: this path differentiates each phase in
        // isolation (split at every return to rest), so a rest-to-rest
        // phase's endpoints have zero measured velocity either side of
        // the whole excursion, whereas the chart search differentiates
        // one continuous trajectory across the entire animation and so
        // retains the curvature at what would otherwise be a phase
        // boundary. Measured on this file's own test fixtures: the
        // 3-phase sinusoid gives 3.61 rad/s^2 here versus ~13.05 rad/s^2
        // from select_candidate on the same input; the merged-reentry
        // (X,X,Z,Z,Y,Y) shape gives 2.69 here versus ~13.49 there. This
        // divergence is real, systematic, and confined to genuinely
        // "AGI "-narrowed configuration animations by the analyze()-level
        // gate in chacha_analyzer.cpp (an ordinary animation never
        // reaches this function), where a per-phase acceleration reading
        // is the more defensible number anyway: an artist's rest-to-rest
        // ramp is exactly the shape this function measures acceleration
        // over. See Config.AccelerationIsMeasuredPerPhaseNotAcrossPhaseBoundaries
        // in chacha_config_test.cpp, which pins this number so it cannot
        // silently drift.
        c.summary.max_velocity[s]     = ph.summary.max_velocity[0];
        c.summary.max_acceleration[s] = ph.summary.max_acceleration[0];
        c.summary.max_residual_rad    = std::max(c.summary.max_residual_rad,
                                                  ph.summary.max_residual_rad);
        c.summary.worst_conditioning  = -1.0f;   // see close_run: not measured
    }
    return c;
}

} // namespace detail
} // namespace ChaCha
