#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace ChaCha {
namespace detail {

namespace {

float angular_distance(const glm::quat& a, const glm::quat& b)
{
    float d = std::fabs(glm::dot(glm::normalize(a), glm::normalize(b)));
    if (d > 1.0f) d = 1.0f;
    return 2.0f * std::acos(d);
}

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
    return angular_distance(axis_quat(axis, angle), q);
}

float residual_two_dof(const glm::quat& q, int axis0, int axis1, const float a[2])
{
    return angular_distance(axis_quat(axis0, a[0]) * axis_quat(axis1, a[1]), q);
}

// Gauss-Newton on the 3-vector residual r(t) = log( R(t)^-1 * q ), two unknowns.
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
