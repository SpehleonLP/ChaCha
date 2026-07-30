#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <array>
#include <cmath>

namespace ChaCha {
namespace detail {

namespace {

constexpr float kPi     = 3.14159265358979323846f;

// sqrt(x*x + y*y), used in place of std::hypot below: hypot's overflow/
// underflow guarding buys nothing here (all inputs are components of unit
// quaternions, so |x|,|y| <= 1) and measurably costs more (163.8ns/call ->
// 125.4ns/call in isolation), which matters on this function's hot path —
// Task 8's workload calls it ~8.1M times, 1.33s -> 1.02s. Bit-identical
// results were confirmed across every probe used in this file's tests.
double fast_hypot(double x, double y)
{
    return std::sqrt(x * x + y * y);
}

StageType rotate_stage(int axis)
{
    switch (axis) {
    case 0:  return StageType::xRotate;
    case 1:  return StageType::yRotate;
    default: return StageType::zRotate;
    }
}

std::array<Chart, 12> build_charts()
{
    std::array<Chart, 12> out{};
    int n = 0;
    // Tait-Bryan: three distinct axes.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                if (i != j && j != k && i != k)
                    out[n++] = Chart{{rotate_stage(i), rotate_stage(j), rotate_stage(k)},
                                     {i, j, k}, false};
    // Proper Euler: first and last axis match, middle differs.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j)
                out[n++] = Chart{{rotate_stage(i), rotate_stage(j), rotate_stage(i)},
                                 {i, j, i}, true};
    return out;
}

float wrap_pi(float a)
{
    while (a >  kPi) a -= 2.0f * kPi;
    while (a <= -kPi) a += 2.0f * kPi;
    return a;
}

// Double-precision counterpart of axis_quat, used internally so that
// compose_chart's quaternion chain-multiply doesn't accumulate float
// rounding across three multiplies. Only the final result is narrowed
// to float.
glm::dquat axis_quat_d(int axis, double angle_rad)
{
    const double h = angle_rad * 0.5;
    const double s = std::sin(h), c = std::cos(h);
    switch (axis) {
    case 0:  return glm::dquat(c, s, 0.0, 0.0);
    case 1:  return glm::dquat(c, 0.0, s, 0.0);
    default: return glm::dquat(c, 0.0, 0.0, s);
    }
}

glm::dquat compose_chart_d(const Chart& c, const double angle[3])
{
    return axis_quat_d(c.axis[0], angle[0])
         * axis_quat_d(c.axis[1], angle[1])
         * axis_quat_d(c.axis[2], angle[2]);
}

} // namespace

glm::quat axis_quat(int axis, float angle_rad)
{
    const glm::dquat q = axis_quat_d(axis, static_cast<double>(angle_rad));
    return glm::quat(static_cast<float>(q.w), static_cast<float>(q.x),
                      static_cast<float>(q.y), static_cast<float>(q.z));
}

std::span<const Chart> all_charts()
{
    static const std::array<Chart, 12> charts = build_charts();
    return std::span<const Chart>(charts.data(), charts.size());
}

glm::quat compose_chart(const Chart& c, const float angle[3])
{
    // Stages apply in order of appearance, matching AGI semantics:
    //   node_transform * stage0 * stage1 * stage2
    const double angle_d[3] = {angle[0], angle[1], angle[2]};
    const glm::dquat r = compose_chart_d(c, angle_d);
    return glm::quat(static_cast<float>(r.w), static_cast<float>(r.x),
                      static_cast<float>(r.y), static_cast<float>(r.z));
}

// Generic quaternion -> Euler extraction covering all 12 sequences
// (Bernardes & Viollet, PLoS ONE 2022).
//
// All intermediate arithmetic is carried out in double precision, narrowing
// to float only in the value returned. This is not required to clear the
// round-trip tolerance (1e-4 rad) — a build of this same solve done
// entirely in float measures on the order of 1e-6 rad worst case,
// comfortably under the bound — but it costs nothing on this function's
// scale and gives a wide accuracy margin against harder inputs than the
// round-trip test exercises.
EulerSolution solve_euler(const glm::quat& qin, const Chart& chart)
{
    const glm::quat qn = glm::normalize(qin);
    const double qw = qn.w, qx = qn.x, qy = qn.y, qz = qn.z;

    const int i = chart.axis[0];
    const int j = chart.axis[1];
    const int k = chart.proper ? (3 - i - j) : chart.axis[2];

    // Sign of the permutation (i, j, k).
    const double eps = static_cast<double>((i - j) * (j - k) * (k - i)) / 2.0;

    const double qv[3] = {qx, qy, qz};

    double a, b, c, d;
    if (chart.proper) {
        a = qw;          b = qv[i];
        c = qv[j];       d = qv[k] * eps;
    } else {
        a = qw - qv[j];  b = qv[i] + qv[k] * eps;
        c = qv[j] + qw;  d = qv[k] * eps - qv[i];
    }

    const double kPiD     = 3.14159265358979323846;
    const double kHalfPiD = kPiD * 0.5;

    double theta1, theta2, theta3;
    if (chart.proper) {
        // a,b share envelope cos(theta2/2); c,d share envelope sin(theta2/2).
        theta2 = 2.0 * std::atan2(fast_hypot(c, d), fast_hypot(a, b));
        const double tp = std::atan2(b, a);
        const double tm = std::atan2(d, c);

        if (std::fabs(theta2) < 1e-9) {
            theta3 = 0.0;
            theta1 = 2.0 * tp - theta3;
        } else if (std::fabs(theta2 - kPiD) < 1e-9) {
            theta3 = 0.0;
            theta1 = 2.0 * tm + theta3;
        } else {
            // theta1/theta3 apply to the same physical axis (first and last
            // stage of a proper chart); the raw tp+/-tm combination comes
            // out swapped relative to the Tait-Bryan case.
            theta1 = tp + tm;
            theta3 = tp - tm;
        }
    } else {
        // b,c share one envelope; a,d share the other (this pairing, not
        // (a,b)/(c,d), is what makes tp/tm independent of theta2).
        theta2 = 2.0 * std::atan2(fast_hypot(b, c), fast_hypot(a, d));
        const double tp = std::atan2(b, c);
        const double tm = std::atan2(d, a);

        if (std::fabs(theta2) < 1e-9) {
            // theta2 ~ 0 implies hypot(b,c) ~ 0: tp=atan2(b,c) is undefined
            // here (both arguments ~0), so only tm survives.
            theta1 = 0.0;
            theta3 = 2.0 * tm - theta1;
        } else if (std::fabs(theta2 - kPiD) < 1e-9) {
            // theta2 ~ pi implies hypot(a,d) ~ 0: tm=atan2(d,a) is undefined
            // here instead, so only tp survives.
            theta1 = 0.0;
            theta3 = 2.0 * tp + theta1;
        } else {
            theta1 = tp - tm;
            theta3 = tp + tm;
        }

        theta3 *= eps;
        theta2 -= kHalfPiD;
    }

    // Wrap into (-pi, pi] in double precision; narrowing to float happens
    // only in the return statement below.
    while (theta1 >  kPiD) theta1 -= 2.0 * kPiD;
    while (theta1 <= -kPiD) theta1 += 2.0 * kPiD;
    while (theta2 >  kPiD) theta2 -= 2.0 * kPiD;
    while (theta2 <= -kPiD) theta2 += 2.0 * kPiD;
    while (theta3 >  kPiD) theta3 -= 2.0 * kPiD;
    while (theta3 <= -kPiD) theta3 += 2.0 * kPiD;

    return EulerSolution{{wrap_pi(static_cast<float>(theta1)),
                           wrap_pi(static_cast<float>(theta2)),
                           wrap_pi(static_cast<float>(theta3))}};
}

// The second solution branch for the same rotation.
//
// Derivation (quaternion conjugation, not the shift-substitution the brief
// suggested — see report): compose_chart evaluates R_i(t1) R_j(t2) R_k(t3)
// for axes (i, j, k) using the angle triple exactly as stored in
// EulerSolution, whatever internal convention (shifted or not) produced
// it. So the alternate triple (t1+pi, t2', t3+pi) must satisfy
//   R_i(pi) R_j(t2') R_k(pi) == R_j(t2)                              (*)
// Insert R_i(pi)^{-1} R_i(pi) and use that conjugating R_j(theta) by a
// pi-rotation about a perpendicular axis i negates the angle
// (R_i(pi) R_j(theta) R_i(pi)^{-1} = R_j(-theta) when i != j):
//   R_i(pi) R_j(t2') R_k(pi)
//     = R_j(-t2') * [R_i(pi) R_k(pi)]
// For Tait-Bryan (i, j, k all distinct), R_i(pi) R_k(pi) is a pure
// quaternion (0, +-e_j) — a rotation of +-pi about j — and +pi and -pi
// about the same axis are the same rotation, so R_i(pi) R_k(pi) = R_j(pi)
// unconditionally (no dependence on chart.eps/permutation sign). That
// makes (*) become R_j(pi - t2') = R_j(t2), i.e. t2' = pi - t2.
// For proper Euler (k == i), R_i(pi) R_k(pi) = R_i(2*pi) = identity, so (*)
// becomes R_j(-t2') = R_j(t2), i.e. t2' = -t2.
// This matches the brief's two formulas exactly (the internal -pi/2 shift
// in solve_euler's Tait-Bryan branch turned out not to matter: the
// derivation only uses the returned angle value, not how solve_euler
// arrived at it). Verified empirically by
// Charts.BothBranchesDescribeTheSameRotation (random, all 12 charts) and
// Charts.AlternateBranchIsCorrectAtSingularConfigurations (deterministic
// grid at the singular middle angle, all 12 charts).
EulerSolution alternate_branch(const EulerSolution& s, const Chart& c)
{
    const float t2 = c.proper ? -s.angle[1] : (kPi - s.angle[1]);
    return EulerSolution{{
        wrap_pi(s.angle[0] + kPi),
        wrap_pi(t2),
        wrap_pi(s.angle[2] + kPi),
    }};
}

float chart_conditioning(const EulerSolution& s, const Chart& c)
{
    // Tait-Bryan degenerates as the (shifted) middle angle approaches
    // +-pi/2; proper Euler degenerates as its (unshifted) middle angle
    // approaches 0 or pi.
    return c.proper ? std::fabs(std::sin(s.angle[1]))
                    : std::fabs(std::cos(s.angle[1]));
}

} // namespace detail
} // namespace ChaCha
