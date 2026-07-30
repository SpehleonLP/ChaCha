#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace ChaCha {
namespace detail {

// Angular distance between two unit quaternions, measured via the relative
// rotation's own log map rather than acos(|dot|). acos is ill-conditioned
// near zero error (its derivative blows up as the argument approaches 1),
// which leaves small residuals reading as exactly 0.0 well before the true
// error actually reaches zero. atan2(|vec|, |w|) has no such singularity at
// the identity and remains well conditioned across the whole range that
// matters here. Task 2 established the same substitution for its own
// round-trip metric, dropping its noise floor from ~9e-4 rad to ~2-3e-7 rad.
//
// Exposed (not file-local) because chacha_search.cpp's evaluate_chart needs
// the same well-conditioned metric to measure a chart candidate's real
// round-trip residual instead of assuming it is exactly zero.
float quaternion_angular_distance(const glm::quat& a, const glm::quat& b)
{
    const glm::quat rel = glm::inverse(glm::normalize(a)) * glm::normalize(b);
    const glm::vec3 v(rel.x, rel.y, rel.z);
    return 2.0f * std::atan2(glm::length(v), std::fabs(rel.w));
}

namespace {

// Log map of a unit quaternion into the tangent space at identity.
glm::vec3 log_map(glm::quat q)
{
    if (q.w < 0.0f) q = -q;             // shortest arc
    const glm::vec3 v(q.x, q.y, q.z);
    const float vn = glm::length(v);
    if (vn < 1e-8f) return glm::vec3(0.0f);
    const float angle = 2.0f * std::atan2(vn, q.w);
    return v * (angle / vn);
}

} // namespace

// The nearest rotation about `axis` to q is its projection onto that axis.
//
// q and -q are the same rotation (unit quaternions doubly cover SO(3)), but
// atan2 is not invariant under negating both arguments the way this
// extraction needs it to be: for a quaternion whose w component happens to
// be negative, atan2(qv[axis], q.w) lands in a different branch of atan2's
// (-pi, pi] range than atan2(-qv[axis], -q.w) does, even though both
// represent the identical rotation. The two results differ by roughly 2pi
// after doubling. Normalizing to the w >= 0 representative first picks the
// same branch regardless of the input's arbitrary sign, matching the
// (-pi, pi] range that axis_quat itself produces.
float solve_one_dof(const glm::quat& q, int axis)
{
    glm::quat qn = q;
    if (qn.w < 0.0f) qn = -qn;
    const float qv[3] = {qn.x, qn.y, qn.z};
    return 2.0f * std::atan2(qv[axis], qn.w);
}

float residual_one_dof(const glm::quat& q, int axis, float angle)
{
    return quaternion_angular_distance(axis_quat(axis, angle), q);
}

float residual_two_dof(const glm::quat& q, int axis0, int axis1, const float a[2])
{
    return quaternion_angular_distance(axis_quat(axis0, a[0]) * axis_quat(axis1, a[1]), q);
}

// Gauss-Newton on the 3-vector residual r(t) = log( R(t)^-1 * q ), two unknowns.
//
// A cold {0,0} seed occasionally stalls at the 24-iteration cap rather than
// diverging: measured over 50000 random targets, restarting the solve from
// its own output improved 1788 of them (3.6%), by up to 0.665 rad. This is a
// stall, not divergence or oscillation, and it only ever occurs on targets
// already far outside the ~0.02 rad accept/reject gate Task 8 uses, so it
// never flips an accept/reject decision. No behaviour change made for this;
// noted here so a future reader doesn't mistake it for a correctness bug.
//
// Warm-seeded cost, measured against Task 8's actual frame-to-frame usage
// pattern (not the random-target benchmark above): only around ~0.7-0.8 us
// per call, close to the single-iteration cost, when BOTH the seed is warm
// AND the target motion is genuinely well explained by this specific
// (axis0, axis1) ordered pair -- both of this function's early exits
// (residual norm < 1e-7, step size < 1e-7) fire almost immediately.
// Otherwise -- a warm seed against motion this ordered pair does not fit
// well, which is the common case, since Task 8's select_candidate tries
// all 6 ordered pairs and by construction 5 of them are "wrong" for any
// given joint -- NEITHER exit condition is ever satisfied and the solve
// runs close to the full 24-iteration cap on every single frame regardless
// of warm-seeding, at roughly 7-7.2 us per call (about 10x the well-fit
// case, not the roughly-14x-cheaper number a reader might assume warm
// seeding always buys). This is the dominant cost in select_candidate's
// two-DOF phase in the common case where no 2-axis pair actually fits.
void solve_two_dof(const glm::quat& q, int axis0, int axis1,
                   const float seed[2], float out[2])
{
    out[0] = seed[0];
    out[1] = seed[1];

    const float kStep = 1e-3f;

    for (int iter = 0; iter < 24; ++iter) {
        auto residual_at = [&](float a0, float a1) {
            const glm::quat r = axis_quat(axis0, a0) * axis_quat(axis1, a1);
            return log_map(glm::inverse(r) * q);
        };

        const glm::vec3 r0 = residual_at(out[0], out[1]);
        if (glm::length(r0) < 1e-7f) break;

        // Numerical Jacobian, 3x2.
        const glm::vec3 c0 = (residual_at(out[0] + kStep, out[1]) - r0) / kStep;
        const glm::vec3 c1 = (residual_at(out[0], out[1] + kStep) - r0) / kStep;

        // Normal equations for a 2x2 system, with light damping for stability.
        const float a = glm::dot(c0, c0) + 1e-6f;
        const float b = glm::dot(c0, c1);
        const float d = glm::dot(c1, c1) + 1e-6f;
        const float g0 = glm::dot(c0, r0);
        const float g1 = glm::dot(c1, r0);

        const float det = a * d - b * b;
        if (std::fabs(det) < 1e-12f) break;

        const float s0 = ( d * g0 - b * g1) / det;
        const float s1 = (-b * g0 + a * g1) / det;

        out[0] -= s0;
        out[1] -= s1;

        if (std::fabs(s0) < 1e-7f && std::fabs(s1) < 1e-7f) break;
    }
}

} // namespace detail
} // namespace ChaCha
