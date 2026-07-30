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
struct OpenRun {
    bool                active{false};
    int                 axis{-1};
    std::vector<float>  times;
    std::vector<float>  angles;
    float               worst_residual{0.0f};
};

// Closes `run` (if any) into `order`, applying the re-entry rule: an axis
// that has already been seen merges into its existing AxisPhase entry
// (first-encounter order preserved) UNLESS a different axis has become
// the most-recently-added entry since -- that is genuine interleaving
// (e.g. X, Z, X) and is rejected. A phase that simply passes back through
// rest at its own midpoint and resumes the same axis (e.g. X, X) is not
// interleaving and merges cleanly.
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
    t.time              = run.times;
    t.angle[0]           = run.angles;
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

// Walks one animation's rest-relative rotation samples, splitting into
// per-axis runs at every return to rest and merging same-axis runs back
// together per the re-entry rule in close_run. Returns false the moment
// either (a) a frame's rotation isn't well explained by any single axis
// (two axes genuinely simultaneous), or (b) an axis's run is interrupted
// mid-phase by a different axis becoming dominant without a return to
// rest, or (c) close_run finds genuine interleaving across runs.
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

    for (size_t i = 0; i < n; ++i) {
        const glm::quat q     = glm::normalize(rel[i]);
        const float     total = total_rotation_angle(q);

        if (total < options.rotation_threshold_rad) {
            if (run.active) {
                // Anchor the end of the sweep at exactly 0 too.
                run.times.push_back(times[i]);
                run.angles.push_back(0.0f);
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
            }
        }

        run.times.push_back(times[i]);
        run.angles.push_back(best_angle);
        run.worst_residual = std::max(run.worst_residual, best_residual);
    }

    return close_run(run, order, options);
}

// Extracts the ordered, merged per-axis phases across every animation in
// `motion`. `order` persists across animations so cross-clip interleaving
// (a joint exercised by more than one AGI animation) is still caught, but
// each animation's open run is force-closed at its own end rather than
// carried across into the next animation's timeline.
bool extract_phases(const JointMotion& motion, const Options& options,
                     std::vector<AxisPhase>& order)
{
    order.clear();
    for (size_t ai = 0; ai < motion.rel_by_animation.size(); ++ai) {
        if (!accumulate_animation(motion.rel_by_animation[ai],
                                   motion.times_by_animation[ai],
                                   options, order)) {
            return false;
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
    if (!extract_phases(motion, options, order) || order.empty()) return c;   // dof 0: locked joint

    c.dof = static_cast<int>(std::min<size_t>(order.size(), 3));
    for (int s = 0; s < c.dof; ++s) {
        const AxisPhase& ph = order[s];
        c.stage[s] = rotate_stage_for(ph.axis);
        c.axis[s]  = ph.axis;

        c.summary.valid               = true;
        c.summary.axis_valid[s]       = true;
        c.summary.min_value[s]        = ph.summary.min_value[0];
        c.summary.max_value[s]        = ph.summary.max_value[0];
        // Velocity and acceleration here reflect the sweep speed the
        // artist authored (usually a uniform ramp), not a physically
        // measured limit -- but that is exactly what every other observed
        // value out of this library means too (see CLAUDE.md: "ChaCha
        // reports observed ROM, not possible ROM"). Report them for
        // consistency with the rest of the pipeline; range and stage
        // order remain the trustworthy outputs of this path specifically.
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
