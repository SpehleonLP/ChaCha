#include <gtest/gtest.h>
#include "chacha_internal.h"
#include "chacha_types.h"
#include <cmath>

using namespace ChaCha;
using namespace ChaCha::detail;

static Trajectory ramp(float from, float to, int n, float duration)
{
    Trajectory t;
    t.time.resize(n);
    for (int a = 0; a < 3; ++a) t.angle[a].assign(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float f = (n == 1) ? 0.0f : static_cast<float>(i) / (n - 1);
        t.time[i]     = f * duration;
        t.angle[0][i] = from + f * (to - from);
    }
    return t;
}

// A sinusoid of known amplitude/frequency has genuine, non-zero acceleration
// everywhere (unlike a straight ramp, whose true acceleration is zero and
// which a broken/no-op implementation could satisfy by accident). Sampling
// it at very different keyframe densities and differentiating the RAW
// keyframes directly (no resample step) gives density-dependent answers:
// sparse keyframes reconstruct a piecewise-linear approximation of the sine
// whose secant slopes underestimate the true peak curvature, while dense
// keyframes approach it closely. Only resampling onto a uniform time grid
// before differentiating recovers the same (correct) peak acceleration
// regardless of how the input was originally keyed.
static Trajectory sinusoid(float amplitude, float angular_freq_rad_s, int n, float duration)
{
    Trajectory t;
    t.time.resize(n);
    for (int a = 0; a < 3; ++a) t.angle[a].assign(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float f = (n == 1) ? 0.0f : static_cast<float>(i) / (n - 1);
        float time = f * duration;
        t.time[i]     = time;
        t.angle[0][i] = amplitude * std::sin(angular_freq_rad_s * time);
    }
    return t;
}

// Ranges must come from observed samples, never seeded at zero.
TEST(Summary, RangeIsSeededFromObservedSamplesNotZero)
{
    Trajectory t = ramp(0.5f, 1.0f, 21, 1.0f);
    CandidateSummary s = summarise(t, 0.0f, Options{});
    EXPECT_NEAR(s.min_value[0], 0.5f, 1e-4f);
    EXPECT_NEAR(s.max_value[0], 1.0f, 1e-4f);
}

TEST(Summary, VelocityMatchesAConstantRamp)
{
    // 1.0 rad over 2.0 s => 0.5 rad/s.
    Trajectory t = ramp(0.0f, 1.0f, 121, 2.0f);
    CandidateSummary s = summarise(t, 0.0f, Options{});
    EXPECT_NEAR(s.max_velocity[0], 0.5f, 0.05f);
}

// See `sinusoid()` above for why a sinusoid (rather than the brief's ramp,
// whose true acceleration is zero everywhere) is the honest signal for this
// claim. Peak acceleration of A*sin(w*t) is A*w^2.
TEST(Summary, AccelerationIsStableUnderKeyframeDensity)
{
    const float amplitude = 1.0f;
    const float w = 2.0f * 3.14159265f;   // one cycle per second
    const float duration = 2.0f;          // two full cycles
    const float true_peak = amplitude * w * w;

    Trajectory sparse = sinusoid(amplitude, w, 61,  duration);
    Trajectory dense  = sinusoid(amplitude, w, 801, duration);

    CandidateSummary a = summarise(sparse, 0.0f, Options{});
    CandidateSummary b = summarise(dense,  0.0f, Options{});

    // Both recover close to the same, genuinely non-zero, true peak
    // acceleration despite the ~13x difference in keyframe density.
    EXPECT_NEAR(a.max_acceleration[0], true_peak, true_peak * 0.15f);
    EXPECT_NEAR(b.max_acceleration[0], true_peak, true_peak * 0.15f);
    EXPECT_NEAR(a.max_acceleration[0], b.max_acceleration[0], true_peak * 0.15f);
}

TEST(Summary, UnionTakesWidestRangeAndWorstConditioning)
{
    CandidateSummary a = summarise(ramp(0.0f, 1.0f, 21, 1.0f), 0.01f, Options{});
    CandidateSummary b = summarise(ramp(-2.0f, 0.5f, 21, 1.0f), 0.03f, Options{});
    a.worst_conditioning = 0.9f;
    b.worst_conditioning = 0.4f;
    union_into(a, b);
    EXPECT_NEAR(a.min_value[0], -2.0f, 1e-4f);
    EXPECT_NEAR(a.max_value[0],  1.0f, 1e-4f);
    EXPECT_NEAR(a.max_residual_rad, 0.03f, 1e-6f);
    EXPECT_NEAR(a.worst_conditioning, 0.4f, 1e-6f);
}
