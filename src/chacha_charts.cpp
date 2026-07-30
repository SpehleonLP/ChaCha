#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <array>
#include <cmath>
#include <limits>

namespace ChaCha {
namespace detail {

namespace {

constexpr float kPi     = 3.14159265358979323846f;
constexpr float kHalfPi = kPi * 0.5f;

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

// The exact angular-distance metric a caller (and this solver's own
// round-trip test) uses to compare two quaternions: float normalize on
// both sides, float dot, clamp, float acos. `target` is expected to be the
// *raw*, not-yet-normalized quaternion a caller holds — normalizing it here
// (once) is what a comparison against it will also do. Passing an
// already-normalized quaternion in would apply a second, redundant
// normalize pass, which is not a no-op: float normalize() is not perfectly
// idempotent, so a second pass can land one ULP away from the first,
// silently comparing against the wrong reference point.
float float_angular_distance(const glm::quat& target, const glm::quat& cur)
{
    float d = std::fabs(glm::dot(glm::normalize(target), glm::normalize(cur)));
    if (d > 1.0f) d = 1.0f;
    return 2.0f * std::acos(d);
}

// Polish the closed-form Euler solution so that, once narrowed to float,
// it reconstructs the input quaternion as closely as possible.
//
// The closed-form solve above is already accurate to roughly float
// epsilon (~1e-7 rad) in double precision, which sounds sufficient — but
// EulerSolution.angle and compose_chart's signature are both `float`, and
// compose_chart itself narrows its double-computed quaternion back to
// float before returning. So the function this solver actually needs to
// invert is the doubly-rounded "float angle -> compose_chart -> float
// quat", not the pure double-precision math above. Because that function
// only changes value at float-representable steps (a staircase), a
// derivative-based method is the wrong tool for the last mile: a small
// finite-difference step can straddle zero, one, or several staircase
// steps depending on where the base value sits.
//
// So instead this does a direct, bounded search of the float grid around
// the closed-form estimate: for each of the three angles, try the float
// itself plus its neighbors a few ULPs in either direction, evaluate every
// combination through the exact float path compose_chart will take, and
// keep whichever combination minimizes float_angular_distance against the
// target quaternion. The search radius (kUlpRadius) trades how far the
// closed-form estimate may be from the true best float against the
// (kUlpRadius*2+1)^3 evaluations this costs.
void refine_euler(const Chart& chart, const glm::quat& target, double theta[3])
{
    constexpr int kUlpRadius = 6;
    constexpr int kSteps = kUlpRadius * 2 + 1;

    const float base[3] = {static_cast<float>(theta[0]),
                            static_cast<float>(theta[1]),
                            static_cast<float>(theta[2])};

    float candidates[3][kSteps];
    for (int k = 0; k < 3; ++k) {
        float v = base[k];
        for (int s = 0; s < kUlpRadius; ++s)
            v = std::nextafter(v, -std::numeric_limits<float>::infinity());
        for (int s = 0; s < kSteps; ++s) {
            candidates[k][s] = v;
            v = std::nextafter(v, std::numeric_limits<float>::infinity());
        }
    }

    float best[3] = {base[0], base[1], base[2]};
    float best_err = float_angular_distance(target, compose_chart(chart, base));

    for (int a = 0; a < kSteps; ++a)
        for (int b = 0; b < kSteps; ++b)
            for (int c = 0; c < kSteps; ++c) {
                const float trial[3] = {candidates[0][a], candidates[1][b], candidates[2][c]};
                const float err = float_angular_distance(target, compose_chart(chart, trial));
                if (err < best_err) {
                    best_err = err;
                    best[0] = candidates[0][a];
                    best[1] = candidates[1][b];
                    best[2] = candidates[2][c];
                }
            }

    theta[0] = best[0];
    theta[1] = best[1];
    theta[2] = best[2];
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
// All intermediate arithmetic is carried out in double precision: the
// round-trip tolerance (1e-4 rad) required of this solver is tighter than
// what chained single-precision trig calls (atan2/hypot/acos) can reliably
// deliver, so we widen internally and narrow back to float only on return.
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
        theta2 = 2.0 * std::atan2(std::hypot(c, d), std::hypot(a, b));
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
        theta2 = 2.0 * std::atan2(std::hypot(b, c), std::hypot(a, d));
        const double tp = std::atan2(b, c);
        const double tm = std::atan2(d, a);

        if (std::fabs(theta2) < 1e-9) {
            theta1 = 0.0;
            theta3 = 2.0 * tp - theta1;
        } else if (std::fabs(theta2 - kPiD) < 1e-9) {
            theta1 = 0.0;
            theta3 = 2.0 * tm + theta1;
        } else {
            theta1 = tp - tm;
            theta3 = tp + tm;
        }

        theta3 *= eps;
        theta2 -= kHalfPiD;
    }

    // Wrap into (-pi, pi] in double precision *before* refining: refine_euler
    // searches the float neighborhood of its input, so the value handed to
    // it must already be what will (modulo exact float rounding) come out
    // the other end — otherwise wrap_pi's own float-precision subtraction
    // could shift the chosen candidate off the neighborhood it was picked
    // from.
    while (theta1 >  kPiD) theta1 -= 2.0 * kPiD;
    while (theta1 <= -kPiD) theta1 += 2.0 * kPiD;
    while (theta2 >  kPiD) theta2 -= 2.0 * kPiD;
    while (theta2 <= -kPiD) theta2 += 2.0 * kPiD;
    while (theta3 >  kPiD) theta3 -= 2.0 * kPiD;
    while (theta3 <= -kPiD) theta3 += 2.0 * kPiD;

    {
        // Refine against the raw, un-normalized `qin`, not `qn`: a caller
        // comparing this solver's output against its own quaternion will
        // normalize that raw value itself (see float_angular_distance's
        // comment on why that must be a single pass, not two).
        double theta[3] = {theta1, theta2, theta3};
        refine_euler(chart, qin, theta);
        theta1 = theta[0];
        theta2 = theta[1];
        theta3 = theta[2];
    }

    return EulerSolution{{wrap_pi(static_cast<float>(theta1)),
                           wrap_pi(static_cast<float>(theta2)),
                           wrap_pi(static_cast<float>(theta3))}};
}

} // namespace detail
} // namespace ChaCha
