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
//
// This is NOT a universal density-independence claim: below roughly 25 Hz
// of keyframes (against this test's 60 Hz resample_rate_hz / window 5
// default), linear interpolation between sparse keyframes introduces
// corners whose curvature resampling cannot remove, and the error is large
// and non-monotonic (measured: 10 Hz keys +38%, 20 Hz +9.8% against the
// analytic peak). 61 and 801 samples over this 2 s trajectory correspond to
// 30 Hz and 400 Hz keyframe rates, both comfortably above that floor, where
// the estimate is stable to within a couple of percent — hence the tight
// (5%) tolerance below, which is what actually pins the window-averaging
// semantics (a window-halving mutation misses this bound by several points).
TEST(Summary, AccelerationIsStableAboveAMinimumKeyframeRate)
{
    const float amplitude = 1.0f;
    const float w = 2.0f * 3.14159265f;   // one cycle per second
    const float duration = 2.0f;          // two full cycles
    const float true_peak = amplitude * w * w;

    Trajectory sparse = sinusoid(amplitude, w, 61,  duration);  // 30 Hz keys
    Trajectory dense  = sinusoid(amplitude, w, 801, duration);  // 400 Hz keys

    CandidateSummary a = summarise(sparse, 0.0f, Options{});
    CandidateSummary b = summarise(dense,  0.0f, Options{});

    // Both recover close to the same, genuinely non-zero, true peak
    // acceleration despite the ~13x difference in keyframe density.
    EXPECT_NEAR(a.max_acceleration[0], true_peak, true_peak * 0.05f);
    EXPECT_NEAR(b.max_acceleration[0], true_peak, true_peak * 0.05f);
    EXPECT_NEAR(a.max_acceleration[0], b.max_acceleration[0], true_peak * 0.05f);
}

// The task's central contract: resampling onto a uniform grid before
// differentiating must make the velocity/acceleration estimate independent
// of how irregularly the input was keyed, not just how densely. This ramp
// mixes very tightly packed keys (0, .02 .. .10 s) with very sparse,
// unevenly spaced ones (.6, 1.2, 1.7, 2.0 s); true velocity is a constant
// 0.5 rad/s and true acceleration is 0 throughout. Differentiating the raw
// keyframes directly with an assumed fixed dt (i.e. skipping resample)
// gives vel=1.0688, accel=1.1517 — both badly wrong — while going through
// resample recovers the correct constant velocity and near-zero
// acceleration.
TEST(Summary, VelocityAndAccelerationAreCorrectUnderNonUniformKeyframeSpacing)
{
    Trajectory t;
    t.time = {0.0f, 0.02f, 0.04f, 0.06f, 0.08f, 0.10f, 0.6f, 1.2f, 1.7f, 2.0f};
    for (int a = 0; a < 3; ++a) t.angle[a].assign(t.time.size(), 0.0f);
    for (size_t i = 0; i < t.time.size(); ++i)
        t.angle[0][i] = 0.5f * t.time[i];

    CandidateSummary s = summarise(t, 0.0f, Options{});
    EXPECT_NEAR(s.max_velocity[0], 0.5f, 0.02f);
    EXPECT_NEAR(s.max_acceleration[0], 0.0f, 0.02f);
}

// The last, possibly sub-`dt`, slice of a trajectory's motion must not be
// silently dropped from the derivative estimate just because the total
// duration isn't an exact multiple of the resample grid spacing. This
// trajectory idles at 0 until t=1.0s, then ramps sharply to 1.0 over the
// final 0.05s (duration=1.05s; at the 10 Hz resample rate used here,
// 1.05*10=10.5 is not integral, so a naive floor(duration*rate) grid stops
// at t=1.0 and never sees the ramp at all, reporting velocity 0 instead of
// a sharp final-interval spike).
TEST(Summary, FinalSubDtIntervalIsNotDroppedFromTheDerivativeEstimate)
{
    Trajectory t;
    t.time = {0.0f, 1.0f, 1.05f};
    t.angle[0] = {0.0f, 0.0f, 1.0f};
    t.angle[1].clear();
    t.angle[2].clear();

    Options o;
    o.resample_rate_hz  = 10.0f;
    o.derivative_window = 3;

    CandidateSummary s = summarise(t, 0.0f, o);
    // The naive (endpoint-dropping) grid reports 0.0 here; a grid that
    // reaches the true final sample reports a clear, non-zero spike.
    EXPECT_GT(s.max_velocity[0], 2.0f);
}

// A per-axis array shorter than `time` (as Task 10's translation/scale
// extraction code may produce for degenerate channels) must not be read
// out of bounds. Verified separately under AddressSanitizer during
// development of this fix (heap-buffer-overflow before, clean after); this
// assertion checks the observable symptom (garbage/non-finite output) that
// a plain, non-sanitized build can still detect.
TEST(Summary, MismatchedAxisArrayLengthDoesNotReadOutOfBounds)
{
    Trajectory t;
    t.time = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    t.angle[0] = {0.0f, 1.0f};   // shorter than t.time
    t.angle[1].clear();
    t.angle[2].clear();

    CandidateSummary s = summarise(t, 0.0f, Options{});
    EXPECT_TRUE(std::isfinite(s.min_value[0]));
    EXPECT_TRUE(std::isfinite(s.max_value[0]));
    EXPECT_TRUE(std::isfinite(s.max_velocity[0]));
    EXPECT_TRUE(std::isfinite(s.max_acceleration[0]));
}

// An axis with no observed samples must not participate in union_into's
// min/max at all — its zero-initialised fields are not a real range and
// must not drag a genuinely-observed range on the other side toward zero.
TEST(Summary, UnionIgnoresAxesWithNoSamplesRatherThanZeroSeedingThem)
{
    CandidateSummary dst;
    dst.valid          = true;
    dst.axis_valid[1]  = true;
    dst.min_value[1]   = 2.0f;
    dst.max_value[1]   = 3.0f;

    CandidateSummary src;
    src.valid         = true;
    src.axis_valid[1] = false; // no samples at all on this axis

    union_into(dst, src);

    EXPECT_NEAR(dst.min_value[1], 2.0f, 1e-6f);
    EXPECT_NEAR(dst.max_value[1], 3.0f, 1e-6f);
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
