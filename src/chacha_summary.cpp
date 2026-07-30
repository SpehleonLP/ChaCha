#include "chacha_internal.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ChaCha {
namespace detail {

namespace {

// Linear resample of an already-continuous angle track onto a uniform grid.
std::vector<float> resample(const std::vector<float>& time,
                            const std::vector<float>& value,
                            float rate_hz)
{
    std::vector<float> out;
    const int n = static_cast<int>(time.size());
    if (n == 0 || rate_hz <= 0.0f) return out;
    if (n == 1) { out.push_back(value[0]); return out; }

    const float dt       = 1.0f / rate_hz;
    const float duration = time[n - 1] - time[0];
    if (duration <= 0.0f) { out.push_back(value[0]); return out; }

    const int steps = static_cast<int>(duration * rate_hz) + 1;
    out.reserve(steps);

    int k = 0;
    for (int i = 0; i < steps; ++i) {
        const float t = time[0] + i * dt;
        while (k + 2 < n && time[k + 1] < t) ++k;
        const float span = time[k + 1] - time[k];
        const float f    = (span > 0.0f) ? std::clamp((t - time[k]) / span, 0.0f, 1.0f) : 0.0f;
        out.push_back(value[k] + f * (value[k + 1] - value[k]));
    }
    return out;
}

// Central difference over a symmetric window; returns max |derivative|.
float max_abs_central_difference(const std::vector<float>& v, float dt, int window)
{
    const int n = static_cast<int>(v.size());
    const int h = std::max(1, window / 2);
    if (n <= 2 * h) return 0.0f;

    float worst = 0.0f;
    for (int i = h; i + h < n; ++i) {
        const float d = (v[i + h] - v[i - h]) / (2.0f * h * dt);
        worst = std::max(worst, std::fabs(d));
    }
    return worst;
}

std::vector<float> central_difference(const std::vector<float>& v, float dt, int window)
{
    const int n = static_cast<int>(v.size());
    const int h = std::max(1, window / 2);
    std::vector<float> out;
    if (n <= 2 * h) return out;
    out.reserve(n - 2 * h);
    for (int i = h; i + h < n; ++i)
        out.push_back((v[i + h] - v[i - h]) / (2.0f * h * dt));
    return out;
}

} // namespace

CandidateSummary summarise(const Trajectory& t, float residual_rad, const Options& options)
{
    CandidateSummary s;
    if (t.time.empty()) return s;

    s.valid              = true;
    s.max_residual_rad   = residual_rad;
    s.worst_conditioning = t.worst_conditioning;

    const float dt = 1.0f / options.resample_rate_hz;

    for (int a = 0; a < 3; ++a) {
        const auto& v = t.angle[a];
        if (v.empty()) continue;

        s.min_value[a] = *std::min_element(v.begin(), v.end());
        s.max_value[a] = *std::max_element(v.begin(), v.end());

        // Derivatives are taken on a uniform grid. Differentiating raw keyframes
        // of a piecewise-linear channel yields impulses whose magnitude scales
        // with keyframe density rather than with motion.
        const std::vector<float> grid = resample(t.time, v, options.resample_rate_hz);
        s.max_velocity[a]     = max_abs_central_difference(grid, dt, options.derivative_window);
        const std::vector<float> vel = central_difference(grid, dt, options.derivative_window);
        s.max_acceleration[a] = max_abs_central_difference(vel, dt, options.derivative_window);
    }
    return s;
}

void union_into(CandidateSummary& dst, const CandidateSummary& src)
{
    if (!src.valid) return;
    if (!dst.valid) { dst = src; return; }

    for (int a = 0; a < 3; ++a) {
        dst.min_value[a]        = std::min(dst.min_value[a], src.min_value[a]);
        dst.max_value[a]        = std::max(dst.max_value[a], src.max_value[a]);
        dst.max_velocity[a]     = std::max(dst.max_velocity[a], src.max_velocity[a]);
        dst.max_acceleration[a] = std::max(dst.max_acceleration[a], src.max_acceleration[a]);
    }
    dst.max_residual_rad   = std::max(dst.max_residual_rad, src.max_residual_rad);
    dst.worst_conditioning = std::min(dst.worst_conditioning, src.worst_conditioning);
}

} // namespace detail
} // namespace ChaCha
